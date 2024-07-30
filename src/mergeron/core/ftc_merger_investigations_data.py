"""
Methods to parse FTC Merger Investigations Data, downloading source documents
as necessary

NOTES
-----
Reported row and column totals from source data are not stored.

"""

import shutil
from collections.abc import Mapping, Sequence
from importlib import resources
from operator import itemgetter
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple, TypeAlias

import msgpack  # type: ignore
import msgpack_numpy as m  # type: ignore
import numpy as np
import re2 as re  # type: ignore
import urllib3
from bs4 import BeautifulSoup
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from .. import _PKG_NAME, DATA_DIR, VERSION  # noqa: TID252

__version__ = VERSION

m.patch()

FTCDATA_DIR = DATA_DIR / "FTCData"
if not FTCDATA_DIR.is_dir():
    FTCDATA_DIR.mkdir(parents=True)

INVDATA_ARCHIVE_PATH = DATA_DIR / "ftc_invdata.msgpack"
if (
    not INVDATA_ARCHIVE_PATH.is_file()
    and (
        _bundled_copy := resources.files(f"{_PKG_NAME}.data").joinpath(
            INVDATA_ARCHIVE_PATH.name
        )
    ).is_file()
):
    with resources.as_file(_bundled_copy) as _bundled_copy_path:
        shutil.copy2(_bundled_copy_path, INVDATA_ARCHIVE_PATH)

TABLE_NO_RE = re.compile(r"Table \d+\.\d+")
TABLE_TYPES = ("ByHHIandDelta", "ByFirmCount")
CONC_TABLE_ALL = "Table 3.1"
CNT_TABLE_ALL = "Table 4.1"

TTL_KEY = 86825
CONC_HHI_DICT = {
    "0 - 1,799": 0,
    "1,800 - 1,999": 1800,
    "2,000 - 2,399": 2000,
    "2,400 - 2,999": 2400,
    "3,000 - 3,999": 3000,
    "4,000 - 4,999": 4000,
    "5,000 - 6,999": 5000,
    "7,000 +": 7000,
    "TOTAL": TTL_KEY,
}
CONC_DELTA_DICT = {
    "0 - 100": 0,
    "100 - 200": 100,
    "200 - 300": 200,
    "300 - 500": 300,
    "500 - 800": 500,
    "800 - 1,200": 800,
    "1,200 - 2,500": 1200,
    "2,500 +": 2500,
    "TOTAL": TTL_KEY,
}
CNT_FCOUNT_DICT = {
    "2 to 1": 2,
    "3 to 2": 3,
    "4 to 3": 4,
    "5 to 4": 5,
    "6 to 5": 6,
    "7 to 6": 7,
    "8 to 7": 8,
    "9 to 8": 9,
    "10 to 9": 10,
    "10 +": 11,
    "TOTAL": TTL_KEY,
}


class INVTableData(NamedTuple):
    industry_group: str
    additional_evidence: str
    data_array: NDArray[np.int64]


INVData: TypeAlias = Mapping[str, dict[str, dict[str, INVTableData]]]


def construct_data(
    _archive_path: Path | None = None,
    *,
    flag_backward_compatibility: bool = True,
    flag_pharma_for_exclusion: bool = True,
    rebuild_data: bool = False,
) -> INVData:
    """Construct FTC merger investigations data for non-overlapping periods,
    from reported data on cumulative periods.

    FTC merger investigations data are reported in cumulative periods,
    e.g., 1996-2003 and 1996-2011, but the analyst may want data reported in
    non-overlapping periods, e.g., 2004-2011. Given the way in which FTC had
    reported merger investigations data, the above example is the only instance
    in which the 1996-2003 data can be subtracted from the cumulative data to
    extract merger investigations data for the later period.
    See also, Kwoka [1]_, Sec. 2.3.3.

    .. [1] Kwoka, J., Greenfield, D., & Gu, C. (2015). Mergers, merger control,
       and remedies: A retrospective analysis of U.S. policy. MIT Press.

    Parameters
    ----------
    _archive_path
        Path to file container for serialized constructed data
    flag_backward_compatibility
        Flag whether the reported data should be treated as backward-compatible
    flag_pharma_for_exclusion
        Flag whether data for Pharmaceuticals is included in,  the set of
        industry groups with consistent reporting in both early and late periods

    Returns
    -------
        A dictionary of merger investigations data keyed to reporting periods

    """
    _archive_path = _archive_path or INVDATA_ARCHIVE_PATH
    if _archive_path.is_file() and not rebuild_data:
        _archived_data = msgpack.unpackb(_archive_path.read_bytes(), use_list=False)

        _invdata: dict[str, dict[str, dict[str, INVTableData]]] = {}
        for _period in _archived_data:
            _invdata[_period] = {}
            for _table_type in _archived_data[_period]:
                _invdata[_period][_table_type] = {}
                for _table_no in _archived_data[_period][_table_type]:
                    _invdata[_period][_table_type][_table_no] = INVTableData(
                        *_archived_data[_period][_table_type][_table_no]
                    )
        return MappingProxyType(_invdata)

    _invdata = dict(_parse_invdata())  # Convert immutable to mutable

    # Add some data periods (
    #   only periods ending in 2011, others have few observations and
    #   some incompatibilities
    #   )
    for _data_period in "2004-2011", "2006-2011", "2008-2011":
        _invdata_bld = _construct_new_period_data(
            _invdata,
            _data_period,
            flag_backward_compatibility=flag_backward_compatibility,
        )
        _invdata |= {_data_period: _invdata_bld}

    # Create data for industries with no evidence on entry
    for _data_period in _invdata:
        _construct_no_entry_evidence_data(_invdata, _data_period)

    # Create a list of exclusions to named industries in the base period,
    #   for construction of aggregate enforcement statistics where feasible
    _industry_exclusion_list = (
        "AllMarkets",
        "OtherMarkets",
        "IndustriesinCommon",
        "",
        ("PharmaceuticalsMarkets" if flag_pharma_for_exclusion else None),
    )
    for _data_period in "1996-2003", "1996-2011", "2004-2011":
        for _table_type, _table_no in zip(
            TABLE_TYPES, (CONC_TABLE_ALL, CNT_TABLE_ALL), strict=True
        ):
            _invdata_sub_tabletype = _invdata[_data_period][_table_type]

            _aggr_tables_list = [
                _t
                for _t in _invdata["1996-2003"][_table_type]
                if re.sub(
                    r"\W", "", _invdata["1996-2003"][_table_type][_t].industry_group
                )
                not in _industry_exclusion_list
            ]

            _invdata_sub_tabletype |= {
                _table_no.replace(".1", ".X"): _invdata_build_aggregate_table(
                    _invdata_sub_tabletype, _aggr_tables_list
                )
            }

    _ = INVDATA_ARCHIVE_PATH.write_bytes(msgpack.packb(_invdata))

    return MappingProxyType(_invdata)


def _construct_no_entry_evidence_data(_invdata: INVData, _data_period: str, /) -> None:
    _invdata_ind_grp = "All Markets"
    _invdata_evid_cond = "No Entry Evidence"

    _invdata_sub_evid_cond_conc = _invdata[_data_period]["ByHHIandDelta"]
    _invdata_sub_evid_cond_conc["Table 9.X"] = INVTableData(
        _invdata_ind_grp,
        _invdata_evid_cond,
        np.column_stack((
            _invdata_sub_evid_cond_conc["Table 3.1"].data_array[:, :2],
            (
                _invdata_sub_evid_cond_conc["Table 3.1"].data_array[:, 2:]
                - _invdata_sub_evid_cond_conc["Table 9.1"].data_array[:, 2:]
                - _invdata_sub_evid_cond_conc["Table 9.2"].data_array[:, 2:]
            ),
        )),
    )

    _invdata_sub_evid_cond_fcount = _invdata[_data_period]["ByFirmCount"]
    _invdata_sub_evid_cond_fcount["Table 10.X"] = INVTableData(
        _invdata_ind_grp,
        _invdata_evid_cond,
        np.column_stack((
            _invdata_sub_evid_cond_fcount["Table 4.1"].data_array[:, :1],
            (
                _invdata_sub_evid_cond_fcount["Table 4.1"].data_array[:, 1:]
                - _invdata_sub_evid_cond_fcount["Table 10.1"].data_array[:, 1:]
                - _invdata_sub_evid_cond_fcount["Table 10.2"].data_array[:, 1:]
            ),
        )),
    )


def _construct_new_period_data(
    _invdata: INVData,
    _data_period: str,
    /,
    *,
    flag_backward_compatibility: bool = False,
) -> dict[str, dict[str, INVTableData]]:
    _cuml_period = "1996-{}".format(int(_data_period.split("-")[1]))
    if _cuml_period != "1996-2011":
        raise ValueError('Expected cumulative period, "1996-2011"')

    _invdata_cuml = _invdata[_cuml_period]

    _base_period = "1996-{}".format(int(_data_period.split("-")[0]) - 1)
    _invdata_base = _invdata[_base_period]

    if tuple(_invdata_cuml.keys()) != TABLE_TYPES:
        raise ValueError("Source data does not include the expected groups of tables.")

    _invdata_bld = {}
    for _table_type in TABLE_TYPES:
        _data_typesubdict = {}
        for _table_no in _invdata_cuml[_table_type]:
            _invdata_cuml_sub_table = _invdata_cuml[_table_type][_table_no]
            _invdata_ind_group, _invdata_evid_cond, _invdata_cuml_array = (
                _invdata_cuml_sub_table.industry_group,
                _invdata_cuml_sub_table.additional_evidence,
                _invdata_cuml_sub_table.data_array,
            )

            _invdata_base_sub_table = _invdata_base[_table_type].get(_table_no, None)

            (_invdata_base_ind_group, _invdata_base_evid_cond, _invdata_base_array) = (
                _invdata_base_sub_table or ("", "", None)
            )

            # Some tables can't be constructed due to inconsistencies in the data
            # across time periods
            if (
                (_data_period != "2004-2011" and _invdata_ind_group != "All Markets")
                or (_invdata_ind_group in ('"Other" Markets', "Industries in Common"))
                or (_invdata_base_ind_group in ('"Other" Markets', ""))
            ):
                continue

            # NOTE: Clean data to enforce consistency in FTC data
            if flag_backward_compatibility:
                # Consistency here means that the number of investigations reported
                # in each period is no less than the number reported in
                # any prior period.Although the time periods for table 3.2 through 3.5
                # are not the samein the data for 1996-2005 and 1996-2007 as in
                # the data for the other periods, they are nonetheless shorter than
                # the period 1996-2011, and hence the counts reported for 1996-2011
                # cannot be less than those reported in these prior periods. Note that
                # The number of "revisions" applied below, for enforcing consistency,
                # is sufficiently small as to be unlikely to substantially impact
                # results from analysis of the data.
                _invdata_cuml_array_stack = []
                _invdata_base_array_stack = []

                for _data_period_detail in _invdata:
                    _pd_start, _pd_end = (
                        int(g) for g in _data_period_detail.split("-")
                    )
                    if _pd_start == 1996:
                        _invdata_cuml_array_stack += [
                            _invdata[_data_period_detail][_table_type][
                                _table_no
                            ].data_array[:, -3:-1]
                        ]
                    if _pd_start == 1996 and _pd_end < int(_data_period.split("-")[0]):
                        _invdata_base_array_stack += [
                            _invdata[_data_period_detail][_table_type][
                                _table_no
                            ].data_array[:, -3:-1]
                        ]
                _invdata_cuml_array_enfcls, _invdata_base_array_enfcls = (
                    np.stack(_f).max(axis=0)
                    for _f in (_invdata_cuml_array_stack, _invdata_base_array_stack)
                )
                _invdata_array_bld_enfcls = (
                    _invdata_cuml_array_enfcls - _invdata_base_array_enfcls
                )
            else:
                # Consistency here means that the most recent data are considered
                # the most accurate, and when constructing data for a new period
                # any negative counts for merger investigations "enforced" or "closed"
                # are reset to zero (non-negativity). The above convention is adopted
                # on the basis of discussions with FTC staff, and given that FTC does
                # not assert backward compatibility published data on
                # merger investigations. Also, FTC appears to maintain that
                # the most recently published data are considered the most accurate
                # account of the pattern of FTC investigations of horizontal mergers,
                # and that the figures for any reported period represent the most
                # accurate data for that period. The published data may not be fully
                # backward compatible due to minor variation in (applying) the criteria
                # for inclusion, as well as industry coding, undertaken to maintain
                # transparency on the enforcement process.
                _invdata_array_bld_enfcls = (
                    _invdata_cuml_array[:, -3:-1] - _invdata_base_array[:, -3:-1]  # type: ignore
                )

                # To examine the number of corrected values per table,
                # uncomment the statements below
                # _invdata_array_bld_tbc = where(
                #   _invdata_array_bld_enfcls < 0, _invdata_array_bld_enfcls, 0
                # )
                # if np.einsum('ij->', invdata_array_bld_tbc):
                #     print(
                #       f"{_data_period}, {_table_no}, {_invdata_ind_group}:",
                #       abs(np.einsum('ij->', invdata_array_bld_tbc))
                #       )

                # Enforce non-negativity
                _invdata_array_bld_enfcls = np.stack((
                    _invdata_array_bld_enfcls,
                    np.zeros_like(_invdata_array_bld_enfcls),
                )).max(axis=0)

            _invdata_array_bld = np.column_stack((
                _invdata_cuml_array[:, :-3],
                _invdata_array_bld_enfcls,
                np.einsum("ij->i", _invdata_array_bld_enfcls),
            ))

            _data_typesubdict[_table_no] = INVTableData(
                _invdata_ind_group, _invdata_evid_cond, _invdata_array_bld
            )
            del _invdata_ind_group, _invdata_evid_cond, _invdata_cuml_array
            del _invdata_base_ind_group, _invdata_base_evid_cond, _invdata_base_array
            del _invdata_array_bld
        _invdata_bld[_table_type] = _data_typesubdict
    return _invdata_bld


def _invdata_build_aggregate_table(
    _data_typesub: dict[str, INVTableData], _aggr_table_list: Sequence[str]
) -> INVTableData:
    _hdr_table_no = _aggr_table_list[0]

    return INVTableData(
        "Industries in Common",
        "Unrestricted on additional evidence",
        np.column_stack((
            _data_typesub[_hdr_table_no].data_array[:, :-3],
            np.einsum(
                "ijk->jk",
                np.stack([
                    (_data_typesub[_t]).data_array[:, -3:] for _t in _aggr_table_list
                ]),
            ),
        )),
    )


def _parse_invdata() -> INVData:
    """Parse FTC merger investigations data reports to structured data.

    Returns
    -------
        Immutable dictionary of merger investigations data, keyed to
        reporting period, and including all tables organized by
        Firm Count (number of remaining competitors) and
        by range of HHI and ∆HHI.

    """
    raise ValueError(
        "This function is defined here as documentation.\n"
        "NOTE: License for `pymupdf`, upon which this function depends,"
        " may be incompatible with the MIT license,"
        " under which this pacakge is distributed."
        " Making this fumction operable requires the user to modify"
        " the source code as well as to install an additional package"
        " not distributed with this package or included in its dependencies."
    )
    import fitz  # type: ignore

    _invdata_docnames = _download_invdata(FTCDATA_DIR)

    _invdata: dict[str, dict[str, dict[str, INVTableData]]] = {}

    for _invdata_docname in _invdata_docnames:
        _invdata_pdf_path = FTCDATA_DIR.joinpath(_invdata_docname)

        _invdata_fitz = fitz.open(_invdata_pdf_path)
        _invdata_meta = _invdata_fitz.metadata
        if _invdata_meta["title"] == " ":
            _invdata_meta["title"] = ", ".join((
                "Horizontal Merger Investigation Data",
                "Fiscal Years",
                "1996-2005",
            ))

        _data_period = re.findall(r"(\d{4}) *(-) *(\d{4})", _invdata_meta["title"])[0]
        _data_period = "".join(_data_period)

        # Initialize containers for parsed data
        _invdata[_data_period] = {k: {} for k in TABLE_TYPES}

        for _pdf_pg in _invdata_fitz.pages():
            _doc_pg_blocks = _pdf_pg.get_text("blocks", sort=False)
            # Across all published reports of FTC investigations data,
            #   sorting lines (PDF page blocks) by the lower coordinates
            #   and then the left coordinates is most effective for
            #   ordering table rows in top-to-bottom order; this doesn't
            #   work for the 1996-2005 data, however, so we resort later
            _doc_pg_blocks = sorted([
                (f"{_f[3]:03.0f}{_f[0]:03.0f}{_f[1]:03.0f}{_f[2]:03.0f}", *_f)
                for _f in _doc_pg_blocks
                if _f[-1] == 0
            ])

            _data_blocks: list[tuple[str]] = [("",)]
            # Pages layouts not the same in all reports
            _pg_hdr_strings = (
                "FEDERAL TRADE COMMISSION",
                "HORIZONTAL MERGER INVESTIGATION DATA: FISCAL YEARS 1996 - 2011",
            )
            if len(_doc_pg_blocks) > 4:
                _tnum: re.match = None
                for _blk_idx, _pg_blk in enumerate(_doc_pg_blocks):
                    if _tnum := TABLE_NO_RE.fullmatch(_pg_blk[-3].strip()):
                        _data_blocks = [
                            _b
                            for _b in _doc_pg_blocks
                            if not _b[-3].startswith(_pg_hdr_strings)
                            and (
                                _b[-3].strip()
                                not in ("Significant Competitors", "Post Merger HHI")
                            )
                            and not re.fullmatch(r"\d+", _b[-3].strip())
                        ]
                        break
                if not _tnum:
                    continue
                del _tnum
            else:
                continue

            _parse_page_blocks(_invdata, _data_period, _data_blocks)

        _invdata_fitz.close()

    return MappingProxyType(_invdata)


def _parse_page_blocks(
    _invdata: INVData, _data_period: str, _doc_pg_blocks: Sequence[Sequence[Any]], /
) -> None:
    if _data_period != "1996-2011":
        _parse_table_blocks(_invdata, _data_period, _doc_pg_blocks)
    else:
        _test_list = [
            (g, f[-3].strip())
            for g, f in enumerate(_doc_pg_blocks)
            if TABLE_NO_RE.fullmatch(f[-3].strip())
        ]
        # In the 1996-2011 report, there are 2 tables per page
        if len(_test_list) == 1:
            _table_a_blocks = _doc_pg_blocks
            _table_b_blocks: Sequence[Sequence[Any]] = []
        else:
            _table_a_blocks, _table_b_blocks = (
                _doc_pg_blocks[_test_list[0][0] : _test_list[1][0]],
                _doc_pg_blocks[_test_list[1][0] :],
            )

        for _table_i_blocks in _table_a_blocks, _table_b_blocks:
            if not _table_i_blocks:
                continue
            _parse_table_blocks(_invdata, _data_period, _table_i_blocks)


def _parse_table_blocks(
    _invdata: INVData, _data_period: str, _table_blocks: Sequence[Sequence[str]], /
) -> None:
    _invdata_evid_cond = "Unrestricted on additional evidence"
    _table_num, _table_ser, _table_type = _identify_table_type(
        _table_blocks[0][-3].strip()
    )

    if _data_period == "1996-2011":
        _invdata_ind_group = (
            _table_blocks[1][-3].split("\n")[1]
            if _table_num == "Table 4.8"
            else _table_blocks[2][-3].split("\n")[0]
        )

        if _table_ser > 4:
            _invdata_evid_cond = (
                _table_blocks[2][-3].split("\n")[1]
                if _table_ser in (9, 10)
                else _table_blocks[3][-3].strip()
            )

    elif _data_period == "1996-2005":
        _table_blocks = sorted(_table_blocks, key=itemgetter(6))

        _invdata_ind_group = _table_blocks[3][-3].strip()
        if _table_ser > 4:
            _invdata_evid_cond = _table_blocks[5][-3].strip()

    elif _table_ser % 2 == 0:
        _invdata_ind_group = _table_blocks[1][-3].split("\n")[2]
        if (_evid_cond_teststr := _table_blocks[2][-3].strip()) == "Outcome":
            _invdata_evid_cond = "Unrestricted on additional evidence"
        else:
            _invdata_evid_cond = _evid_cond_teststr

    elif _table_blocks[3][-3].startswith("FTC Horizontal Merger Investigations"):
        _invdata_ind_group = _table_blocks[3][-3].split("\n")[2]
        _invdata_evid_cond = "Unrestricted on additional evidence"

    else:
        # print(_table_blocks)
        _invdata_evid_cond = (
            _table_blocks[1][-3].strip()
            if _table_ser == 9
            else _table_blocks[3][-3].strip()
        )
        _invdata_ind_group = _table_blocks[4][-3].split("\n")[2]

    if _invdata_ind_group == "Pharmaceutical Markets":
        _invdata_ind_group = "Pharmaceuticals Markets"

    process_table_func = (
        _process_table_blks_conc_type
        if _table_type == TABLE_TYPES[0]
        else _process_table_blks_cnt_type
    )

    _table_array = process_table_func(_table_blocks)
    if not isinstance(_table_array, np.ndarray) or _table_array.dtype != np.int64:
        print(_table_num)
        print(_table_blocks)
        raise ValueError

    _table_data = INVTableData(_invdata_ind_group, _invdata_evid_cond, _table_array)
    _invdata[_data_period][_table_type] |= {_table_num: _table_data}


def _identify_table_type(_tnstr: str = CONC_TABLE_ALL, /) -> tuple[str, int, str]:
    _tnum = _tnstr.split(" ")[1]
    _tsub = int(_tnum.split(".")[0])
    return _tnstr, _tsub, TABLE_TYPES[(_tsub + 1) % 2]


def _process_table_blks_conc_type(
    _table_blocks: Sequence[Sequence[str]], /
) -> NDArray[np.int64]:
    _conc_row_pat = re.compile(r"((?:0|\d,\d{3}) (?:- \d+,\d{3}|\+)|TOTAL)")

    _col_titles_array = tuple(CONC_DELTA_DICT.values())
    _col_totals: NDArray[np.int64] = np.zeros(len(_col_titles_array), np.int64)
    _invdata_array: NDArray[np.int64] = np.array(None)

    for _tbl_blk in _table_blocks:
        if _conc_row_pat.match(_blk_str := _tbl_blk[-3]):
            _row_list: list[str] = _blk_str.strip().split("\n")
            _row_title: str = _row_list.pop(0)
            _row_key: int = CONC_HHI_DICT[_row_title]
            _row_total = np.array(_row_list.pop().replace(",", "").split("/"), np.int64)
            _row_array_list: list[list[int]] = []
            while _row_list:
                _enfd_val, _clsd_val = _row_list.pop(0).split("/")
                _row_array_list += [
                    [
                        _row_key,
                        _col_titles_array[len(_row_array_list)],
                        int(_enfd_val),
                        int(_clsd_val),
                        int(_enfd_val) + int(_clsd_val),
                    ]
                ]
            _row_array = np.array(_row_array_list, np.int64)
            # Check row totals
            assert_array_equal(_row_total, np.einsum("ij->j", _row_array[:, 2:4]))

            if _row_key == TTL_KEY:
                _col_totals = _row_array
            else:
                _invdata_array = (
                    np.vstack((_invdata_array, _row_array))
                    if _invdata_array.shape
                    else _row_array
                )
            del _row_array, _row_array_list
        else:
            continue

    # Check column totals
    for _col_tot in _col_totals:
        assert_array_equal(
            _col_tot[2:],
            np.einsum(
                "ij->j", _invdata_array[_invdata_array[:, 1] == _col_tot[1]][:, 2:]
            ),
        )

    return _invdata_array[
        np.argsort(np.einsum("ij,ij->i", [[100, 1]], _invdata_array[:, :2]))
    ]


def _process_table_blks_cnt_type(
    _table_blocks: Sequence[Sequence[str]], /
) -> NDArray[np.int64]:
    _cnt_row_pat = re.compile(r"(\d+ (?:to \d+|\+)|TOTAL)")

    _invdata_array: NDArray[np.int64] = np.array(None)
    _col_totals: NDArray[np.int64] = np.zeros(
        3, np.int64
    )  # "enforced", "closed", "total"

    for _tbl_blk in _table_blocks:
        if _cnt_row_pat.match(_blk_str := _tbl_blk[-3]):
            _row_list_s = _blk_str.strip().replace(",", "").split("\n")
            _row_list = np.array(
                [CNT_FCOUNT_DICT[_row_list_s[0]], *_row_list_s[1:]], np.int64
            )
            del _row_list_s
            if _row_list[3] != _row_list[1] + _row_list[2]:
                raise ValueError(
                    "Total number of investigations does not equal #enforced plus #closed."
                )
            if _row_list[0] == TTL_KEY:
                _col_totals = _row_list
            else:
                _invdata_array = (
                    np.vstack((_invdata_array, _row_list))
                    if _invdata_array.shape
                    else _row_list
                )
        else:
            continue

    if not np.array_equal(
        np.array([int(f) for f in _col_totals[1:]], np.int64),
        np.einsum("ij->j", _invdata_array[:, 1:]),
    ):
        raise ValueError("Column totals don't compute.")

    return _invdata_array[np.argsort(_invdata_array[:, 0])]


def _download_invdata(_dl_path: Path = FTCDATA_DIR) -> tuple[str, ...]:
    if not _dl_path.is_dir():
        _dl_path.mkdir(parents=True)

    _invdata_homepage_urls = (
        "https://www.ftc.gov/reports/horizontal-merger-investigation-data-fiscal-years-1996-2003",
        "https://www.ftc.gov/reports/horizontal-merger-investigation-data-fiscal-years-1996-2005-0",
        "https://www.ftc.gov/reports/horizontal-merger-investigation-data-fiscal-years-1996-2007-0",
        "https://www.ftc.gov/reports/horizontal-merger-investigation-data-fiscal-years-1996-2011",
    )
    _invdata_docnames = (
        "040831horizmergersdata96-03.pdf",
        "p035603horizmergerinvestigationdata1996-2005.pdf",
        "081201hsrmergerdata.pdf",
        "130104horizontalmergerreport.pdf",
    )

    if all(
        _dl_path.joinpath(_invdata_docname).is_file()
        for _invdata_docname in _invdata_docnames
    ):
        return _invdata_docnames

    _invdata_docnames_dl: tuple[str, ...] = ()
    _u3pm = urllib3.PoolManager()
    _chunk_size = 1024 * 1024
    for _invdata_homepage_url in _invdata_homepage_urls:
        with _u3pm.request(
            "GET", _invdata_homepage_url, preload_content=False
        ) as _u3handle:
            _invdata_soup = BeautifulSoup(_u3handle.data, "html.parser")
            _invdata_attrs = [
                (_g.get("title", ""), _g.get("href", ""))
                for _g in _invdata_soup.find_all("a")
                if _g.get("title", "") and _g.get("href", "").endswith(".pdf")
            ]
        for _invdata_attr in _invdata_attrs:
            _invdata_docname, _invdata_link = _invdata_attr
            _invdata_docnames_dl += (_invdata_docname,)
            with (
                _u3pm.request(
                    "GET", f"https://www.ftc.gov/{_invdata_link}", preload_content=False
                ) as _urlopen_handle,
                _dl_path.joinpath(_invdata_docname).open("wb") as _invdata_fh,
            ):
                while True:
                    _data = _urlopen_handle.read(_chunk_size)
                    if not _data:
                        break
                    _invdata_fh.write(_data)

    return _invdata_docnames_dl


if __name__ == "__main__":
    print(
        "This module defines functions for downloading and preparing FTC merger investigations data for further analysis."
    )
