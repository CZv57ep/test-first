"""
Functions to parse FTC Merger Investigations Data, downloading source documents
as necessary.

NOTE: We avoid use of reported totals by range of HHI and delta from
source data, to avoid potential minor inconsistencies.

"""

from collections.abc import Mapping, Sequence
from importlib import metadata
from operator import itemgetter
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import fitz  # type: ignore
import msgpack  # type: ignore
import msgpack_numpy as m  # type: ignore
import numpy as np
import re2 as re  # type: ignore
import requests
from bs4 import BeautifulSoup
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

m.patch()

__version__ = metadata.version(Path(__file__).parents[1].stem)

pkg_path = Path(__file__).parent.parent
data_dir = Path.home() / pkg_path.stem / "FTCData"
if not data_dir.is_dir():
    data_dir.mkdir(parents=True)
invdata_dump_path = data_dir.parent / "ftc_invdata_array_dict.msgpack"

table_no_re = re.compile(r"Table \d+\.\d+")
table_types = ("ByHHIandDelta", "ByFirmCount")
ind_grp_key = "Industry Group"
evid_grp_key = "Additional Evidence"
data_array_key = "Data Detail"
conc_table_all = "Table 3.1"
cnt_table_all = "Table 4.1"

ttl_key = 86825
conchhi_dict = {
    "0 - 1,799": 0,
    "1,800 - 1,999": 1800,
    "2,000 - 2,399": 2000,
    "2,400 - 2,999": 2400,
    "3,000 - 3,999": 3000,
    "4,000 - 4,999": 4000,
    "5,000 - 6,999": 5000,
    "7,000 +": 7000,
    "TOTAL": ttl_key,
}
concdelta_dict = {
    "0 - 100": 0,
    "100 - 200": 100,
    "200 - 300": 200,
    "300 - 500": 300,
    "500 - 800": 500,
    "800 - 1,200": 800,
    "1,200 - 2,500": 1200,
    "2,500 +": 2500,
    "TOTAL": ttl_key,
}
frmcnt_dict = {
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
    "TOTAL": ttl_key,
}


class TableData(NamedTuple):
    ind_grp: str
    evid_cond: str
    data_array: NDArray[np.int_]


def construct_invdata(
    _invdata_dump_path: Path | None = None,
    *,
    flag_backward_compatibility: bool = True,
    flag_pharma_for_exclusion: bool = True,
    rebuild_data: bool = False,
) -> MappingProxyType[str, dict[str, dict[str, TableData]]]:
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
    _invdata_dump_path
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
    _invdata_dump_path = _invdata_dump_path or invdata_dump_path
    if _invdata_dump_path.is_file() and not rebuild_data:
        _invdata_dict_loaded = msgpack.unpackb(
            _invdata_dump_path.read_bytes(), use_list=False
        )

        _invdata_dict: dict[str, dict[str, dict[str, TableData]]] = {}
        for _period in _invdata_dict_loaded:
            _invdata_dict[_period] = {}
            for _table_type in _invdata_dict_loaded[_period]:
                _invdata_dict[_period][_table_type] = {}
                for _table_no in _invdata_dict_loaded[_period][_table_type]:
                    _invdata_dict[_period][_table_type][_table_no] = TableData(
                        *_invdata_dict_loaded[_period][_table_type][_table_no]
                    )
        return MappingProxyType(_invdata_dict)

    _invdata_array_dict = dict(parse_invdata())  # Convert immutable to mutable

    # Add some data periods (
    #   only periods ending in 2011, others have few observations and
    #   some incompatibilities
    #   )
    for _data_period in "2004-2011", "2006-2011", "2008-2011":
        _invdata_array_dict_bld = _construct_new_period_data(
            _invdata_array_dict,
            _data_period,
            flag_backward_compatibility=flag_backward_compatibility,
        )
        _invdata_array_dict |= {_data_period: _invdata_array_dict_bld}

    # Create data for industries with no evidence on entry
    for _data_period in _invdata_array_dict:
        _construct_no_entry_evidence_data(_invdata_array_dict, _data_period)

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
            table_types, (conc_table_all, cnt_table_all), strict=True
        ):
            _dat_array_type_subdict = _invdata_array_dict[_data_period][_table_type]

            _aggr_tables_list = [
                _t
                for _t in _invdata_array_dict["1996-2003"][_table_type]
                if re.sub(
                    r"\W", "", _invdata_array_dict["1996-2003"][_table_type][_t].ind_grp
                )
                not in _industry_exclusion_list
            ]

            _dat_array_type_subdict |= {
                _table_no.replace(".1", ".X"): _invdata_build_aggregate_table(
                    _dat_array_type_subdict, _aggr_tables_list
                )
            }

    _ = invdata_dump_path.write_bytes(msgpack.packb(_invdata_array_dict))

    return MappingProxyType(_invdata_array_dict)


def _construct_no_entry_evidence_data(
    _invdata_array_dict: Mapping[str, dict[str, dict[str, TableData]]],
    _data_period: str,
    /,
) -> None:
    _invdata_ind_grp = "All Markets"
    _invdata_evid_cond = "No Entry Evidence"

    _invdata_evid_cond_sub_conc = _invdata_array_dict[_data_period]["ByHHIandDelta"]
    _invdata_evid_cond_sub_conc["Table 9.X"] = TableData(
        _invdata_ind_grp,
        _invdata_evid_cond,
        np.column_stack((
            _invdata_evid_cond_sub_conc["Table 3.1"].data_array[:, :2],
            (
                _invdata_evid_cond_sub_conc["Table 3.1"].data_array[:, 2:]
                - _invdata_evid_cond_sub_conc["Table 9.1"].data_array[:, 2:]
                - _invdata_evid_cond_sub_conc["Table 9.2"].data_array[:, 2:]
            ),
        )),
    )

    _invdata_evid_cond_sub_fcount = _invdata_array_dict[_data_period]["ByFirmCount"]
    _invdata_evid_cond_sub_fcount["Table 10.X"] = TableData(
        _invdata_ind_grp,
        _invdata_evid_cond,
        np.column_stack((
            _invdata_evid_cond_sub_fcount["Table 4.1"].data_array[:, :1],
            (
                _invdata_evid_cond_sub_fcount["Table 4.1"].data_array[:, 1:]
                - _invdata_evid_cond_sub_fcount["Table 10.1"].data_array[:, 1:]
                - _invdata_evid_cond_sub_fcount["Table 10.2"].data_array[:, 1:]
            ),
        )),
    )


def _construct_new_period_data(
    _invdata_array_dict: Mapping[str, dict[str, dict[str, TableData]]],
    _data_period: str,
    /,
    *,
    flag_backward_compatibility: bool = False,
) -> dict[str, dict[str, TableData]]:
    _cuml_period = "1996-{}".format(int(_data_period.split("-")[1]))
    if _cuml_period != "1996-2011":
        raise ValueError('Expected cumulative period, "1996-2011"')

    _invdata_array_dict_cuml = _invdata_array_dict[_cuml_period]

    _base_period = "1996-{}".format(int(_data_period.split("-")[0]) - 1)
    _invdata_array_dict_base = _invdata_array_dict[_base_period]

    if tuple(_invdata_array_dict_cuml.keys()) != table_types:
        raise ValueError("Source data does not include the expected groups of tables.")

    _invdata_array_dict_bld = {}
    for _table_type in table_types:
        _invdata_typesubdict = {}
        for _table_no in _invdata_array_dict_cuml[_table_type]:
            _invdata_table_cuml = _invdata_array_dict_cuml[_table_type][_table_no]
            _invdata_indugrp, _invdata_evid_cond, _invdata_array_cuml = (
                _invdata_table_cuml.ind_grp,
                _invdata_table_cuml.evid_cond,
                _invdata_table_cuml.data_array,
            )

            _invdata_table_base = _invdata_array_dict_base[_table_type].get(
                _table_no, None
            )

            (_invdata_indugrp_base, _invdata_evid_cond_base, _invdata_array_base) = (
                _invdata_table_base or ("", "", None)
            )

            # Some tables can't be constructed due to inconsistencies in the data
            # across time periods
            if (
                (_data_period != "2004-2011" and _invdata_indugrp != "All Markets")
                or (_invdata_indugrp in ('"Other" Markets', "Industries in Common"))
                or (_invdata_indugrp_base in ('"Other" Markets', ""))
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
                _invdata_array_cuml_stack = []
                _invdata_array_base_stack = []

                for _data_period_detail in _invdata_array_dict:
                    _pd_start, _pd_end = (
                        int(g) for g in _data_period_detail.split("-")
                    )
                    if _pd_start == 1996:
                        _invdata_array_cuml_stack += [
                            _invdata_array_dict[_data_period_detail][_table_type][
                                _table_no
                            ].data_array[:, -3:-1]
                        ]
                    if _pd_start == 1996 and _pd_end < int(_data_period.split("-")[0]):
                        _invdata_array_base_stack += [
                            _invdata_array_dict[_data_period_detail][_table_type][
                                _table_no
                            ].data_array[:, -3:-1]
                        ]
                _invdata_array_cuml_enfcls, _invdata_array_base_enfcls = (
                    np.stack(_f).max(axis=0)
                    for _f in (_invdata_array_cuml_stack, _invdata_array_base_stack)
                )
                _invdata_array_bld_enfcls = (
                    _invdata_array_cuml_enfcls - _invdata_array_base_enfcls
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
                    _invdata_array_cuml[:, -3:-1] - _invdata_array_base[:, -3:-1]  # type: ignore
                )

                # To examine the number of corrected values per table,
                # uncomment the statements below
                # _invdata_array_bld_tbc = where(
                #   _invdata_array_bld_enfcls < 0, _invdata_array_bld_enfcls, 0
                # )
                # if np.einsum('ij->', invdata_array_bld_tbc):
                #     print(
                #       f"{_data_period}, {_table_no}, {_invdata_indugrp}:",
                #       abs(np.einsum('ij->', invdata_array_bld_tbc))
                #       )

                # Enforce non-negativity
                _invdata_array_bld_enfcls = np.stack((
                    _invdata_array_bld_enfcls,
                    np.zeros_like(_invdata_array_bld_enfcls),
                )).max(axis=0)

            _invdata_array_bld = np.column_stack((
                _invdata_array_cuml[:, :-3],
                _invdata_array_bld_enfcls,
                np.einsum("ij->i", _invdata_array_bld_enfcls),
            ))

            _invdata_typesubdict[_table_no] = TableData(
                _invdata_indugrp, _invdata_evid_cond, _invdata_array_bld
            )
            del _invdata_indugrp, _invdata_evid_cond, _invdata_array_cuml
            del _invdata_indugrp_base, _invdata_evid_cond_base, _invdata_array_base
            del _invdata_array_bld
        _invdata_array_dict_bld[_table_type] = _invdata_typesubdict
    return _invdata_array_dict_bld


def _invdata_build_aggregate_table(
    _invdata_array_dict_typesub: dict[str, TableData], _aggr_table_list: Sequence[str]
) -> TableData:
    _hdr_table_no = _aggr_table_list[0]

    return TableData(
        "Industries in Common",
        "Unrestricted on additional evidence",
        np.column_stack((
            _invdata_array_dict_typesub[_hdr_table_no].data_array[:, :-3],
            np.einsum(
                "ijk->jk",
                np.stack([
                    (_invdata_array_dict_typesub[_t]).data_array[:, -3:]
                    for _t in _aggr_table_list
                ]),
            ),
        )),
    )


def parse_invdata(
    _invdata_docnames: Sequence[str] = (
        "040831horizmergersdata96-03.pdf",
        "p035603horizmergerinvestigationdata1996-2005.pdf",
        "081201hsrmergerdata.pdf",
        "130104horizontalmergerreport.pdf",
    ),
) -> MappingProxyType[str, dict[str, dict[str, TableData]]]:
    """Parse FTC merger investigations data reports to structured data.

    Parameters
    ----------
    _invdata_docnames
        Names of PDF files reporting FTC merger investigations data.

    Returns
    -------
        Immutable dictionary of merger investigations data, keyed to
        reporting period, and including all tables organized by
        Firm Count (number of remaining competitors) and
        by range of HHI and ∆HHI.

    """
    _invdata_array_dict: dict[str, dict[str, dict[str, TableData]]] = {}

    for _invdata_docname in _invdata_docnames:
        _invdata_pdf_path = data_dir.joinpath(_invdata_docname)
        if not _invdata_pdf_path.is_file():
            _download_invdata(data_dir)

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
        _invdata_array_dict[_data_period] = {k: {} for k in table_types}

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
                    if _tnum := table_no_re.fullmatch(_pg_blk[-3].strip()):
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

            _parse_page_blocks(_invdata_array_dict, _data_period, _data_blocks)

        _invdata_fitz.close()

    return MappingProxyType(_invdata_array_dict)


def _parse_page_blocks(
    _invdata_array_dict: dict[str, dict[str, dict[str, TableData]]],
    _data_period: str,
    _doc_pg_blocks: Sequence[Sequence[Any]],
    /,
) -> None:
    if _data_period != "1996-2011":
        _parse_table_blocks(_invdata_array_dict, _data_period, _doc_pg_blocks)
    else:
        _test_list = [
            (g, f[-3].strip())
            for g, f in enumerate(_doc_pg_blocks)
            if table_no_re.fullmatch(f[-3].strip())
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
            _parse_table_blocks(_invdata_array_dict, _data_period, _table_i_blocks)


def _parse_table_blocks(
    _invdata_array_dict: dict[str, dict[str, dict[str, TableData]]],
    _data_period: str,
    _table_blocks: Sequence[Sequence[str]],
    /,
) -> None:
    _invdata_evid_cond = "Unrestricted on additional evidence"
    _table_num, _table_ser, _table_type = _identify_table_type(
        _table_blocks[0][-3].strip()
    )

    if _data_period == "1996-2011":
        _invdata_indugrp = (
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

        _invdata_indugrp = _table_blocks[3][-3].strip()
        if _table_ser > 4:
            _invdata_evid_cond = _table_blocks[5][-3].strip()

    elif _table_ser % 2 == 0:
        _invdata_indugrp = _table_blocks[1][-3].split("\n")[2]
        if (_evid_cond_teststr := _table_blocks[2][-3].strip()) == "Outcome":
            _invdata_evid_cond = "Unrestricted on additional evidence"
        else:
            _invdata_evid_cond = _evid_cond_teststr

    elif _table_blocks[3][-3].startswith("FTC Horizontal Merger Investigations"):
        _invdata_indugrp = _table_blocks[3][-3].split("\n")[2]
        _invdata_evid_cond = "Unrestricted on additional evidence"

    else:
        # print(_table_blocks)
        _invdata_evid_cond = (
            _table_blocks[1][-3].strip()
            if _table_ser == 9
            else _table_blocks[3][-3].strip()
        )
        _invdata_indugrp = _table_blocks[4][-3].split("\n")[2]

    if _invdata_indugrp == "Pharmaceutical Markets":
        _invdata_indugrp = "Pharmaceuticals Markets"

    process_table_func = (
        _process_table_blks_conc_type
        if _table_type == table_types[0]
        else _process_table_blks_cnt_type
    )

    _table_array = process_table_func(_table_blocks)
    if not isinstance(_table_array, np.ndarray) or _table_array.dtype != np.int_:
        print(_table_num)
        print(_table_blocks)
        raise ValueError

    _table_data = TableData(_invdata_indugrp, _invdata_evid_cond, _table_array)
    _invdata_array_dict[_data_period][_table_type] |= {_table_num: _table_data}


def _identify_table_type(_tnstr: str = conc_table_all, /) -> tuple[str, int, str]:
    _tnum = _tnstr.split(" ")[1]
    _tsub = int(_tnum.split(".")[0])
    return _tnstr, _tsub, table_types[(_tsub + 1) % 2]


def _process_table_blks_conc_type(
    _table_blocks: Sequence[Sequence[str]], /
) -> NDArray[np.int_]:
    _conc_row_pat = re.compile(r"((?:0|\d,\d{3}) (?:- \d+,\d{3}|\+)|TOTAL)")

    _col_titles_array = tuple(concdelta_dict.values())
    # _col_totals: NDArray[np.int_] | None = None
    _invdata_array: NDArray[np.int_] = np.array(None)

    for _tbl_blk in _table_blocks:
        if _conc_row_pat.match(_blk_str := _tbl_blk[-3]):
            _row_list: list[str] = _blk_str.strip().split("\n")
            _row_title: str = _row_list.pop(0)
            _row_key: int = conchhi_dict[_row_title]
            _row_total = np.array(_row_list.pop().replace(",", "").split("/"), np.int_)
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
            _row_array = np.array(_row_array_list, np.int_)
            # Check row totals
            assert_array_equal(_row_total, np.einsum("ij->j", _row_array[:, 2:4]))

            if _row_key == ttl_key:
                _col_totals = _row_array
            else:
                _invdata_array = (
                    np.row_stack((_invdata_array, _row_array))
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
) -> NDArray[np.int_]:
    _cnt_row_pat = re.compile(r"(\d+ (?:to \d+|\+)|TOTAL)")

    _invdata_array: NDArray[np.int_] = np.array(None)

    for _tbl_blk in _table_blocks:
        if _cnt_row_pat.match(_blk_str := _tbl_blk[-3]):
            _row_list_s = _blk_str.strip().replace(",", "").split("\n")
            _row_list = np.array(
                [frmcnt_dict[_row_list_s[0]], *_row_list_s[1:]], np.int_
            )
            del _row_list_s
            if _row_list[3] != _row_list[1] + _row_list[2]:
                raise ValueError(
                    "Total number of investigations does not equal #enforced plus #closed."
                )
            if ttl_key == _row_list[0]:
                _col_totals = _row_list
            else:
                _invdata_array = (
                    np.row_stack((_invdata_array, _row_list))
                    if _invdata_array.shape
                    else _row_list
                )
        else:
            continue

    if not np.array_equal(
        np.array([int(f) for f in _col_totals[1:]], np.int_),
        np.einsum("ij->j", _invdata_array[:, 1:]),
    ):
        raise ValueError("Column totals don't compute.")

    return _invdata_array[np.argsort(_invdata_array[:, 0])]


def _download_invdata(_dl_path: Path) -> list[Any]:
    _invdata_homepage_urls = (
        "https://www.ftc.gov/reports/horizontal-merger-investigation-data-fiscal-years-1996-2003",
        "https://www.ftc.gov/reports/horizontal-merger-investigation-data-fiscal-years-1996-2005-0",
        "https://www.ftc.gov/reports/horizontal-merger-investigation-data-fiscal-years-1996-2007-0",
        "https://www.ftc.gov/reports/horizontal-merger-investigation-data-fiscal-years-1996-2011",
    )
    _invdata_docnames = []
    for _invdata_homepage_url in _invdata_homepage_urls:
        _invdata_soup = BeautifulSoup(
            requests.get(_invdata_homepage_url, verify=True, timeout=60).text,
            "html.parser",
        )
        _invdata_attrs = [
            (_g.get("href", ""), _g.get("title", ""))
            for _g in _invdata_soup.find_all("a")
            if _g.get("title", "") and _g.get("href", "").endswith(".pdf")
        ]
        for _invdata_attr in _invdata_attrs:
            _invdata_link, _invdata_docname = _invdata_attr
            _invdata_docnames += [_invdata_docname]
            with _dl_path.joinpath(_invdata_docname).open("wb") as _invdata_fh:
                _invdata_fh.write(
                    requests.get(
                        f"https://www.ftc.gov/{_invdata_link}", verify=True, timeout=60
                    ).content
                )

    return _invdata_docnames


if __name__ == "__main__":
    invdata_array_dict = construct_invdata(
        invdata_dump_path,
        flag_backward_compatibility=True,
        flag_pharma_for_exclusion=True,
        rebuild_data=False,
    )

    for data_period in invdata_array_dict:
        print(data_period, "-->")
        for table_type in (isd1 := invdata_array_dict[data_period]):
            leader_str = "\t"
            print(leader_str, table_type, "-->")
            leader_str += "\t"
            for table_no in (isd11 := isd1[table_type]):
                (invdata_indugrp, invdata_evid_cond, table_data_array) = isd11[table_no]
                print(
                    leader_str,
                    table_no,
                    " \u2014 ",
                    invdata_indugrp,
                    f", {invdata_evid_cond or 'N/A'}",
                    ", ",
                    sep="",
                    end="",
                )
                print(
                    "Odds ratio = {}/{}".format(
                        *np.einsum("ij->j", table_data_array[:, -3:])
                    )
                )
        print("\n")

    inv_period, inv_type, inv_table = "2004-2011", "HHI and Delta", "Table 3.3"
    #  inv_period, inv_type, inv_table = "2004-2011", "Firm Count", "Table 4.1"
    print(f"Investigations data, {inv_period}, by {inv_type}, {inv_table}")
    print(
        "{}, {}\n{}".format(
            *invdata_array_dict[inv_period][f"By{inv_type.replace(' ', '')}"][inv_table]
        )
    )
