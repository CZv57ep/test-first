"""
Functions to format summary data on merger enforcement patterns.

"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgpack_numpy as m  # type: ignore
import numpy as np
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from numpy.typing import NDArray
from scipy.interpolate import interp1d  # type: ignore

import mergeron.core.ftc_merger_investigations_data as fid
from mergeron.core.ftc_merger_investigations_data import TableData
from mergeron.core.proportions_tests import propn_ci

m.patch()

__version__ = metadata.version(Path(__file__).parents[1].stem)


@enum.unique
class INDGRPConstants(enum.StrEnum):
    ALL = "All Markets"
    GRO = "Grocery Markets"
    OIL = "Oil Markets"
    CHM = "Chemical Markets"
    PHM = "Pharmaceuticals Markets"
    HOS = "Hospital Markets"
    EDS = "Electronically-Controlled Devices and Systems Markets"
    BRD = "Branded Consumer Goods Markets"
    OTH = '"Other" Markets'
    IIC = "Industries in Common"


@enum.unique
class EVIDENConstants(enum.StrEnum):
    HD = "Hot Documents Identified"
    CC = "Strong Customer Complaints"
    NE = "No Entry Evidence"
    ED = "Entry Difficult"
    UR = "Unrestricted on additional evidence"


@enum.unique
class CLRENFSelector(enum.StrEnum):
    CLRN = "clearance"
    ENFT = "enforcement"
    BOTH = "both"


@enum.unique
class StatsGrpSelector(enum.StrEnum):
    FC = "ByFirmCount"
    HD = "ByHHIandDelta"
    DL = "ByDelta"
    ZN = "ByConcZone"


@enum.unique
class StatsReturnSelector(enum.StrEnum):
    CNT = "count"
    RPT = "rate, point"
    RIN = "rate, interval"


@enum.unique
class SortSelector(enum.StrEnum):
    UCH = "unchanged"
    REV = "reversed"


cnt_format_str = R"{: >5,.0f}"
pct_format_str = R"{: >6.1f}\%"
ci_propn_format_str = R"{0: >5.1f} [{2: >4.1f},{3: >5.1f}] \%"

moe_tmpl = Template(R"""
    {% if (rv[2] - rv[0]) | abs == (rv[3] - rv[0]) | abs %}
         {{- "[\pm {:.1f}]".format(rv[3] - rv[0]) -}}
    {% else %}
         {{- "[{:.1f}/+{:.1f}]".format(rv[2] - rv[0], rv[3] - rv[0]) -}}
    {% endif %}
    """)

ltx_array_lineend = R"\\" "\n"
format_latex_hrdcoldesc = "{}\n{}\n{}".format(
    "".join((
        R"\matrix[hcol, above=0pt of {}, nodes = {{",
        R"text width={}, text depth=10pt, inner sep=3pt, minimum height=25pt,",
        R"}},] ",
        R"({}) ",
        R"{{",
    )),
    R"\node[align = {},] {{ {} }}; \\",
    R"}};",
)

# Define the latex jinja environment
# http://eosrei.net/articles/2015/11/latex-templates-python-and-jinja2-generate-pdfs
latex_jinja_env = Environment(
    block_start_string=R"((*",
    block_end_string="*))",
    variable_start_string=R"\JINVAR{",
    variable_end_string="}",
    comment_start_string=R"((#",  # r'#{',
    comment_end_string=R"#))",  # '}',
    line_statement_prefix="%%",
    line_comment_prefix="%#",
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=select_autoescape(disabled_extensions=("tex.jinja2",)),
    loader=FileSystemLoader(Path(__file__).parents[1] / "jinja_LaTex_templates"),
)


class ClearanceRateDataContainer(SimpleNamespace):
    """A container for passing content to jinja2 templates

    Other attributes added later, to fully populate selected jinja2 templates
    """

    invdata_hdrstr: str
    invdata_datstr: str


# Parameters and functions to interpolate selected HHI and ΔHHI values
#   recorded in fractions to ranges of values in points on the HHI scale
hhi_delta_knots = np.array(
    [0, 100, 200, 300, 500, 800, 1200, 2500, 5001], dtype=np.int_
)
hhi_zone_post_knots = np.array([0, 1800, 2400, 10001], dtype=np.int_)
hhi_delta_ranger, hhi_zone_post_ranger = (
    interp1d(_f / 1e4, _f, kind="previous", assume_sorted=True)
    for _f in (hhi_delta_knots, hhi_zone_post_knots)
)

hmg_presumption_zone_dict = {
    hhi_zone_post_knots[0]: {
        hhi_delta_knots[0]: (0, 0, 0),
        hhi_delta_knots[1]: (0, 0, 0),
        hhi_delta_knots[2]: (0, 0, 0),
    },
    hhi_zone_post_knots[1]: {
        hhi_delta_knots[0]: (0, 1, 1),
        hhi_delta_knots[1]: (1, 1, 2),
        hhi_delta_knots[2]: (1, 1, 2),
    },
    hhi_zone_post_knots[2]: {
        hhi_delta_knots[0]: (0, 2, 1),
        hhi_delta_knots[1]: (1, 2, 3),
        hhi_delta_knots[2]: (2, 2, 4),
    },
}

zone_vals = np.unique(
    np.row_stack([
        tuple(hmg_presumption_zone_dict[_k].values())
        for _k in hmg_presumption_zone_dict
    ]),
    axis=0,
)

zone_str_dict = {
    0: R"\node[align = left, fill=BrightGreen] {Green Zone (Safeharbor)};",
    1: R"\node[align = left, fill=HiCoYellow] {Yellow Zone};",
    2: R"\node[align = left, fill=VibrRed] {Red Zone (SLC Presumption)};",
    fid.ttl_key: R"\node[align = left, fill=OBSHDRFill] {TOTAL};",
}

zone_detail_str_hhi_dict = {
    0: Rf"HHI_{{post}} < \text{{{hhi_zone_post_knots[1]} pts.}}",
    1: R"HHI_{{post}} \in \text{{[{}, {}) pts. and }} ".format(
        *hhi_zone_post_knots[1:3]
    ),
    2: Rf"HHI_{{post}} \geqslant \text{{{hhi_zone_post_knots[2]} pts. and }} ",
}

zone_detail_str_delta_dict = {
    0: "",
    1: Rf"\Delta HHI < \text{{{hhi_delta_knots[1]} pts.}}",
    2: Rf"\Delta HHI \geqslant \text{{{hhi_delta_knots[1]} pts.}}",
    3: R"\Delta HHI \in \text{{[{}, {}) pts.}}".format(*hhi_delta_knots[1:3]),
    4: Rf"\Delta HHI \geqslant \text{{{hhi_delta_knots[2]} pts.}}",
}


def clrenf_stats_output(
    _data_array_dict: Mapping[str, Mapping[Any, Any]],
    _data_period: str = "1996-2003",
    _table_ind_group: INDGRPConstants = INDGRPConstants.ALL,
    _table_evid_cond: EVIDENConstants = EVIDENConstants.UR,
    _stats_group: StatsGrpSelector = StatsGrpSelector.FC,
    _clrenf_sel: CLRENFSelector = CLRENFSelector.CLRN,
    /,
    *,
    return_type_sel: StatsReturnSelector = StatsReturnSelector.RPT,
    sort_order: SortSelector = SortSelector.UCH,
    print_to_screen: bool = True,
) -> tuple[list[str], list[list[str]]]:
    if _data_period not in _data_array_dict:
        raise ValueError(
            f"Value of _data_period, {_data_period!r} is invalid.",
            f"Must be in, {_data_array_dict!r}",
        )

    if _stats_group == StatsGrpSelector.HD:
        raise ValueError(
            'Statistics formatted, "{StatsGrpSelector.HD}" not available here.'
        )
    else:
        _latex_tbl_clrenf_stats_func = {
            StatsGrpSelector.ZN: latex_tbl_clrenf_stats_byzone,
            StatsGrpSelector.FC: latex_tbl_clrenf_stats_1dim,
            StatsGrpSelector.DL: latex_tbl_clrenf_stats_1dim,
        }[_stats_group]

    _clrenf_stats_cnts = clrenf_stats_cnts_by_group(
        _data_array_dict,
        _data_period,
        _table_ind_group,
        _table_evid_cond,
        _stats_group,
        _clrenf_sel,
    )

    _clrenf_stats_hdr_list, _clrenf_stats_dat_list = _latex_tbl_clrenf_stats_func(  # type: ignore
        _clrenf_stats_cnts, None, return_type_sel=return_type_sel, sort_order=sort_order
    )

    if print_to_screen:
        print(
            f"{_clrenf_sel.capitalize()} stats ({return_type_sel})",
            f"for Period: {_data_period}",
            "\u2014",
            f"{_table_ind_group};",
            _table_evid_cond,
        )
        for g in range(len(_clrenf_stats_hdr_list)):
            print(
                _clrenf_stats_hdr_list[g],
                "&",
                " & ".join(_clrenf_stats_dat_list[g]),
                ltx_array_lineend,
                end="",
            )
        print()

    return _clrenf_stats_hdr_list, _clrenf_stats_dat_list


def clrenf_stats_cnts_by_group(
    _invdata_array_dict: Mapping[str, Mapping[str, Mapping[str, TableData]]],
    _study_period: str,
    _table_ind_grp: INDGRPConstants,
    _table_evid_cond: EVIDENConstants,
    _stats_group: StatsGrpSelector,
    _clrenf_sel: CLRENFSelector,
    /,
) -> NDArray[np.int_]:
    if _stats_group == StatsGrpSelector.HD:
        raise ValueError(
            f"Clearance/enforcement statistics, '{_stats_group}' not valied here."
        )

    match _stats_group:
        case StatsGrpSelector.FC:
            _clrenf_cnts_func = clrenf_cnts_byfirmcount
            _clrenf_cnts_listing_func = clrenf_cnts_listing_byfirmcount
        case StatsGrpSelector.DL:
            _clrenf_cnts_func = clrenf_cnts_bydelta
            _clrenf_cnts_listing_func = clrenf_cnts_listing_byhhianddelta
        case StatsGrpSelector.ZN:
            _clrenf_cnts_func = clrenf_cnts_byconczone
            _clrenf_cnts_listing_func = clrenf_cnts_listing_byhhianddelta

    return _clrenf_cnts_func(
        _clrenf_cnts_listing_func(
            _invdata_array_dict,
            _study_period,
            _table_ind_grp,
            _table_evid_cond,
            _clrenf_sel,
        )
    )


def clrenf_cnts_listing_byfirmcount(
    _data_array_dict: Mapping[str, Mapping[str, Mapping[str, TableData]]],
    _data_period: str = "1996-2003",
    _table_ind_group: INDGRPConstants = INDGRPConstants.ALL,
    _table_evid_cond: EVIDENConstants = EVIDENConstants.UR,
    _clrenf_sel: CLRENFSelector = CLRENFSelector.CLRN,
    /,
) -> NDArray[np.int_]:
    if _data_period not in _data_array_dict:
        raise ValueError(
            f"Invalid value of data period, {_data_period!r}."
            f"Must be one of, {tuple(_data_array_dict.keys())!r}."
        )

    _data_array_dict_sub = _data_array_dict[_data_period][fid.table_types[1]]

    _table_no = table_no_lku(_data_array_dict_sub, _table_ind_group, _table_evid_cond)

    _clrenf_cnts_array = _data_array_dict_sub[_table_no].data_array

    _ndim_in = 1
    _stats_kept_indxs = []
    match _clrenf_sel:
        case CLRENFSelector.CLRN:
            _stats_kept_indxs = [-1, -2]
        case CLRENFSelector.ENFT:
            _stats_kept_indxs = [-1, -3]
        case CLRENFSelector.BOTH:
            _stats_kept_indxs = [-1, -3, -2]

    return np.column_stack([
        _clrenf_cnts_array[:, :_ndim_in],
        _clrenf_cnts_array[:, _stats_kept_indxs],
    ])


def clrenf_cnts_listing_byhhianddelta(
    _data_array_dict: Mapping[str, Mapping[str, Mapping[str, TableData]]],
    _data_period: str = "1996-2003",
    _table_ind_group: INDGRPConstants = INDGRPConstants.ALL,
    _table_evid_cond: EVIDENConstants = EVIDENConstants.UR,
    _clrenf_sel: CLRENFSelector = CLRENFSelector.CLRN,
    /,
) -> NDArray[np.int_]:
    if _data_period not in _data_array_dict:
        raise ValueError(
            f"Invalid value of data period, {_data_period!r}."
            f"Must be one of, {tuple(_data_array_dict.keys())!r}."
        )

    _data_array_dict_sub = _data_array_dict[_data_period][fid.table_types[0]]

    _table_no = table_no_lku(_data_array_dict_sub, _table_ind_group, _table_evid_cond)

    _clrenf_cnts_array = _data_array_dict_sub[_table_no].data_array

    _ndim_in = 2
    _stats_kept_indxs = []
    match _clrenf_sel:
        case CLRENFSelector.CLRN:
            _stats_kept_indxs = [-1, -2]
        case CLRENFSelector.ENFT:
            _stats_kept_indxs = [-1, -3]
        case CLRENFSelector.BOTH:
            _stats_kept_indxs = [-1, -3, -2]

    return np.column_stack([
        _clrenf_cnts_array[:, :_ndim_in],
        _clrenf_cnts_array[:, _stats_kept_indxs],
    ])


def table_no_lku(
    _data_array_dict_sub: Mapping[str, TableData],
    _table_ind_group: INDGRPConstants = INDGRPConstants.ALL,
    _table_evid_cond: EVIDENConstants = EVIDENConstants.UR,
    /,
) -> str:
    if _table_ind_group not in (
        _igl := [_data_array_dict_sub[_v].ind_grp for _v in _data_array_dict_sub]
    ):
        raise ValueError(
            f"Invalid value for industry group, {_table_ind_group!r}."
            f"Must be one of {_igl!r}"
        )

    _tno = next(
        _t
        for _t in _data_array_dict_sub
        if all((
            _data_array_dict_sub[_t].ind_grp == _table_ind_group,
            _data_array_dict_sub[_t].evid_cond == _table_evid_cond,
        ))
    )

    return _tno


def clrenf_cnts_byfirmcount(
    _clrenf_cnts_array: NDArray[np.integer], /
) -> NDArray[np.int_]:
    _ndim_in = 1
    return np.row_stack([
        np.concatenate([
            (f,),
            np.einsum(
                "ij->j", _clrenf_cnts_array[_clrenf_cnts_array[:, 0] == f][:, _ndim_in:]
            ),
        ])
        for f in np.unique(_clrenf_cnts_array[:, 0])
    ])


def clrenf_cnts_bydelta(_clrenf_cnts_array: NDArray[np.integer], /) -> NDArray[np.int_]:
    _ndim_in = 2
    return np.row_stack([
        np.concatenate([
            (f,),
            np.einsum(
                "ij->j", _clrenf_cnts_array[_clrenf_cnts_array[:, 1] == f][:, _ndim_in:]
            ),
        ])
        for f in hhi_delta_knots[:-1]
    ])


def clrenf_cnts_byconczone(
    _clrenf_cnts_array: NDArray[np.integer], /
) -> NDArray[np.int_]:
    # Prepare to tag clearance stats by presumption zone
    _hhi_zone_post_ranged = hhi_zone_post_ranger(_clrenf_cnts_array[:, 0] / 1e4)
    _hhi_delta_ranged = hhi_delta_ranger(_clrenf_cnts_array[:, 1] / 1e4)

    # Step 1: Tag and agg. from HHI-post and Delta to zone triple
    # NOTE: Although you could just map and not (partially) aggregate in this step,
    # the mapped array is a copy, and is larger without partial aggregation, so
    # aggregation reduces the footprint of this step in memory. Although this point
    # is more relevant for generated than observed data, using the same coding pattern
    # in both cases does make life easier
    _ndim_in = 2
    _nkeys = 3
    _clrenf_cnts_byhhipostanddelta = -1 * np.ones(
        _nkeys + _clrenf_cnts_array.shape[1] - _ndim_in, dtype=np.int64
    )
    _clrenf_cnts_byconczone = -1 * np.ones_like(_clrenf_cnts_byhhipostanddelta)
    for _hhi_zone_post_lim in hhi_zone_post_knots[:-1]:
        _level_test = _hhi_zone_post_ranged == _hhi_zone_post_lim

        for _hhi_zone_delta_lim in hhi_delta_knots[:3]:
            _delta_test = (
                (_hhi_delta_ranged > hhi_delta_knots[1])
                if _hhi_zone_delta_lim == hhi_delta_knots[2]
                else (_hhi_delta_ranged == _hhi_zone_delta_lim)
            )

            _zone_val = hmg_presumption_zone_dict[_hhi_zone_post_lim][
                _hhi_zone_delta_lim
            ]

            _conc_test = _level_test & _delta_test

            _clrenf_cnts_byhhipostanddelta = np.row_stack((
                _clrenf_cnts_byhhipostanddelta,
                np.array(
                    (
                        *_zone_val,
                        *np.einsum(
                            "ij->j", _clrenf_cnts_array[:, _ndim_in:][_conc_test]
                        ),
                    ),
                    dtype=np.int64,
                ),
            ))
    _clrenf_cnts_byhhipostanddelta = _clrenf_cnts_byhhipostanddelta[1:]

    for _zone_val in zone_vals:
        # Logical-and of multiple vectors:
        _hhi_zone_test = (
            1
            * np.column_stack([
                _clrenf_cnts_byhhipostanddelta[:, _idx] == _val
                for _idx, _val in enumerate(_zone_val)
            ])
        ).prod(axis=1) == 1

        _clrenf_cnts_byconczone = np.row_stack((
            _clrenf_cnts_byconczone,
            np.concatenate(
                (
                    _zone_val,
                    np.einsum(
                        "ij->j",
                        _clrenf_cnts_byhhipostanddelta[_hhi_zone_test][:, _nkeys:],
                    ),
                ),
                dtype=np.int64,
            ),
        ))

    return _clrenf_cnts_byconczone[1:]


def latex_tbl_clrenf_stats_1dim(
    _inparr: NDArray[np.integer] | NDArray[np.floating],
    _totals_row: int | None = None,
    /,
    *,
    return_type_sel: StatsReturnSelector = StatsReturnSelector.CNT,
    sort_order: SortSelector = SortSelector.UCH,
) -> tuple[list[str], list[list[str]]]:
    _ndim_in: int = 1
    _dim_hdr_dict = {
        0: "{[0, 100)}",
        100: "{[100, 200)}",
        200: "{[200, 300)}",
        300: "{[300, 500)}",
        500: "{[500, 800)}",
        800: "{[800, 1200)}",
        1200: "{[1200, 2500)}",
        2500: "{[2500, 5000]}",
        2: "{2 to 1}",
        3: "{3 to 2}",
        4: "{4 to 3}",
        5: "{5 to 4}",
        6: "{6 to 5}",
        7: "{7 to 6}",
        8: "{8 to 7}",
        9: "{9 to 8}",
        10: "{10 to 9}",
        11: "{10 +}",
        fid.ttl_key: "TOTAL",
    }

    if sort_order == SortSelector.REV:
        _inparr = _inparr[::-1]

    if _totals_row is None:
        _inparr = np.row_stack((
            _inparr,
            np.concatenate(([fid.ttl_key], np.einsum("ij->j", _inparr[:, _ndim_in:]))),
        ))

    _stats_hdr_list, _stats_dat_list = [], []
    for _stats_row in _inparr:
        _stats_hdr_list += [_dim_hdr_dict[_stats_row[0]]]

        _stats_cnt = _stats_row[_ndim_in:]
        _stats_tot = np.concatenate((
            [_inparr[-1][_ndim_in]],
            _stats_cnt[0] * np.ones_like(_stats_cnt[1:]),
        ))
        _stats_dat_list += _stats_formatted_row(_stats_cnt, _stats_tot, return_type_sel)

    return _stats_hdr_list, _stats_dat_list


def latex_tbl_clrenf_stats_byzone(
    _inparr: NDArray[np.integer] | NDArray[np.floating],
    _totals_row: int | None = None,
    /,
    *,
    return_type_sel: StatsReturnSelector = StatsReturnSelector.CNT,
    sort_order: SortSelector = SortSelector.UCH,
) -> tuple[list[str], list[list[str]]]:
    _ndim_in: int = zone_vals.shape[1]

    _zone_str_keys = list(zone_str_dict)
    if sort_order == SortSelector.REV:
        _inparr = _inparr[::-1]
        _zone_str_keys = _zone_str_keys[:-1][::-1] + [_zone_str_keys[-1]]

    if _totals_row is None:
        _inparr = np.row_stack((
            _inparr,
            np.concatenate((
                [fid.ttl_key, -1, -1],
                np.einsum("ij->j", _inparr[:, _ndim_in:]),
            )),
        ))

    _stats_hdr_list, _stats_dat_list = ([], [])
    for _conc_zone in _zone_str_keys:
        _stats_byzone_it = _inparr[_inparr[:, 0] == _conc_zone]
        _stats_hdr_list += [zone_str_dict[_conc_zone]]

        _stats_cnt = np.einsum("ij->j", _stats_byzone_it[:, _ndim_in:])
        _stats_tot = np.concatenate((
            [_inparr[-1][_ndim_in]],
            _stats_cnt[0] * np.ones_like(_stats_cnt[1:]),
        ))
        _stats_dat_list += _stats_formatted_row(_stats_cnt, _stats_tot, return_type_sel)

        if _conc_zone in (2, fid.ttl_key):
            continue

        for _stats_byzone_detail in _stats_byzone_it:
            # Only two sets of subtotals detail, so
            # a conditional expression will do here
            _stats_text_color = "HiCoYellow" if _conc_zone == 1 else "BrightGreen"
            _stats_hdr_list += [
                R"{} {{ \({}{}\) }};".format(
                    rf"\node[text = {_stats_text_color}, fill = white, align = right]",
                    zone_detail_str_hhi_dict[_stats_byzone_detail[1]],
                    (
                        ""
                        if _stats_byzone_detail[2] == 0
                        else Rf"{zone_detail_str_delta_dict[_stats_byzone_detail[2]]}"
                    ),
                )
            ]

            _stats_cnt = _stats_byzone_detail[_ndim_in:]
            _stats_tot = np.concatenate((
                [_inparr[-1][_ndim_in]],
                _stats_cnt[0] * np.ones_like(_stats_cnt[1:]),
            ))
            _stats_dat_list += _stats_formatted_row(
                _stats_cnt, _stats_tot, return_type_sel
            )

    return _stats_hdr_list, _stats_dat_list


def _stats_formatted_row(
    _stats_row_cnt: NDArray[np.integer],
    _stats_row_tot: NDArray[np.integer],
    _return_type_sel: StatsReturnSelector,
    /,
) -> list[list[str]]:
    _stats_row_pct = _stats_row_cnt / _stats_row_tot

    if _return_type_sel == StatsReturnSelector.RIN:
        _stats_row_ci = np.array([
            propn_ci(*g, method="Wilson")
            for g in zip(_stats_row_cnt[1:], _stats_row_tot[1:], strict=True)
        ])
        return [
            [
                pct_format_str.format(100 * _stats_row_pct[0]),
                *[
                    ci_propn_format_str.format(*100 * np.array(f)).replace(
                        R"  nan [ nan,  nan] \%", "---"
                    )
                    for f in _stats_row_ci
                ],
            ]
        ]
    elif _return_type_sel == StatsReturnSelector.RPT:
        return [
            [
                pct_format_str.format(f).replace(R"nan\%", "---")
                for f in 100 * _stats_row_pct
            ]
        ]
    else:
        return [
            [cnt_format_str.format(f).replace(R"nan", "---") for f in _stats_row_cnt]
        ]
