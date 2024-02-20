"""

Construct observed clearance rates and enforcement rates, by specification,
from FTC merger investigations data

"""

import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

from numpy import einsum, row_stack, unique

import mergeron.core.ftc_merger_investigations_data as fid
import mergeron.gen.investigations_stats as isl
from mergeron import DATA_DIR

if not sys.warnoptions:
    warnings.simplefilter("ignore")

INVRES_RATIO_FORMAT_STR = "{: >3.0f}/{:<3.0f}"
INVDATA_DOTTEX_FORMAT_STR = "{}.tex".format(
    "_".join(("FTCMergerInvestigationsDataTables", "{}", "OBS"))
)


def invres_stats_odds_ratio_byhhianddelta(
    _data_array_dict: Mapping,
    _data_periods: tuple[str, str],
    _merger_classes: Sequence[isl.INDGRPConstants | isl.EVIDENConstants],
    /,
) -> tuple[str, ...]:
    """
    Reconstruct tables by HHI and Delta.

    Source tables as well as tables from constructed periods.
    """
    if not all(_dpd in _data_array_dict for _dpd in _data_periods):
        raise ValueError(
            f"All given data periods, {_data_periods!r} must be contained "
            f"in {tuple(_data_array_dict.keys())!r}"
        )

    print("Odds ratios by HHI and Delta:")
    _stats_group = isl.StatsGrpSelector.HD
    _invres_rate_table_content = isl.StatsContainer()
    _invres_rate_table_design = isl.latex_jinja_env.get_template(
        "ftcinvdata_byhhianddelta_table_template.tex.jinja2"
    )
    _invres_rate_table_content.obs_summary_type = f"{_stats_group}"

    _output_dottex_pathlist: tuple[str, ...] = ()
    for _merger_class in _merger_classes:
        _table_ind_group = (
            _merger_class
            if isinstance(_merger_class, isl.INDGRPConstants)
            else isl.INDGRPConstants.ALL
        )
        _table_evid_cond = (
            _merger_class
            if isinstance(_merger_class, isl.EVIDENConstants)
            else isl.EVIDENConstants.UR
        )
        _invres_rate_table_content.obs_merger_class = f"{_merger_class}"

        for _data_period in _data_periods:
            _data_array_dict_sub = _data_array_dict[_data_period][f"{_stats_group}"]
            _table_no = isl.table_no_lku(
                _data_array_dict_sub, _table_ind_group, _table_evid_cond
            )

            _invres_rate_table_content.table_ref = _table_no

            _data_period_0, _data_period_1 = (int(f) for f in _data_period.split("-"))
            if _data_period_0 != 1996:
                _invres_rate_table_content.invdata_notestr = " ".join((
                    "NOTES:",
                    isl.LTX_ARRAY_LINEEND,
                    R"\(\cdot\) Data for period, {}".format(
                        _data_periods[1].replace("-", "--")
                    ),
                    f"calculated by subtracting reported figures for 1996--{_data_period_0 - 1}",
                    R"from reported figures for 1996--2011",
                    isl.LTX_ARRAY_LINEEND,
                    isl.LTX_ARRAY_LINEEND,
                ))

                _invres_rate_table_content.invdata_sourcestr = " ".join((
                    "\\(\\cdot\\) Fed. Trade Comm'n ({}), at~\\cref{{fn:{}}},".format(
                        _data_period_0, f"FTCInvData1996to{_data_period_0}"
                    ),
                    isl.LTX_ARRAY_LINEEND,
                ))
                _invres_rate_table_content.invdata_sourcestr += " ".join((
                    "\\(\\cdot\\) Fed. Trade Comm'n ({}), at~\\cref{{fn:{}}},".format(
                        _data_period_1, f"FTCInvData1996to{_data_period_1}"
                    ),
                    isl.LTX_ARRAY_LINEEND,
                ))
            else:
                _invres_rate_table_content.invdata_sourcestr = " ".join((
                    "\\(\\cdot\\) Fed. Trade Comm'n ({}), at~\\cref{{fn:{}}},".format(
                        _data_period_1, f"FTCInvData1996to{_data_period_1}"
                    ),
                    isl.LTX_ARRAY_LINEEND,
                ))

            _invres_rate_table_content.obs_merger_class = f"{_merger_class}"
            _invres_rate_table_content.obs_period = _data_period.split("-")

            _invres_cnts_array = _data_array_dict_sub[_table_no][-1]
            _odds_ratio_data_str = ""
            for _hhi_range_it in unique(_invres_cnts_array[:, 0]):
                _invres_cnts_row_for_hhi_range = _invres_cnts_array[
                    _invres_cnts_array[:, 0] == _hhi_range_it
                ][:, 2:]
                _odds_ratio_data_str += " & ".join([
                    INVRES_RATIO_FORMAT_STR.format(*g)
                    for g in _invres_cnts_row_for_hhi_range
                ])
                _odds_ratio_data_str += " & {}".format(
                    INVRES_RATIO_FORMAT_STR.format(
                        *einsum("ij->j", _invres_cnts_row_for_hhi_range)
                    )
                )
                _odds_ratio_data_str += isl.LTX_ARRAY_LINEEND

            _invres_cnts_row_for_hhi_tots = row_stack([
                einsum(
                    "ij->j", _invres_cnts_array[_invres_cnts_array[:, 1] == f][:, 2:]
                )
                for f in unique(_invres_cnts_array[:, 1])
            ])
            _odds_ratio_data_str += " & ".join([
                INVRES_RATIO_FORMAT_STR.format(*f)
                for f in _invres_cnts_row_for_hhi_tots
            ])
            _odds_ratio_data_str += " & {}".format(
                INVRES_RATIO_FORMAT_STR.format(
                    *einsum("ij->j", _invres_cnts_row_for_hhi_tots)
                )
            )
            _odds_ratio_data_str += isl.LTX_ARRAY_LINEEND
            _invres_rate_table_content.invdata_byhhianddelta = _odds_ratio_data_str

            _output_dottex_path = DATA_DIR / INVDATA_DOTTEX_FORMAT_STR.format(
                f"{_stats_group}_{_data_period_1}_{_merger_class.replace(' ', '')}"
            )
            with _output_dottex_path.open(
                "w", encoding="utf8"
            ) as _invres_rate_table_dottex:
                _invres_rate_table_dottex.write(
                    _invres_rate_table_design.render(
                        tmpl_data=_invres_rate_table_content
                    )
                )
                print("\n", file=_invres_rate_table_dottex)

            _output_dottex_pathlist += (_output_dottex_path.name,)
            print(_odds_ratio_data_str)
            print()
            del _odds_ratio_data_str

    return _output_dottex_pathlist


def invres_stats_obs_setup(
    _data_array_dict: Mapping,
    _data_periods: tuple[str, str],
    _merger_classes: Sequence[isl.INDGRPConstants | isl.EVIDENConstants],
    _test_regime: isl.PolicySelector = isl.PolicySelector.CLRN,
    /,
) -> tuple[str, ...]:
    _notes_str_base = " ".join((
        "NOTES:",
        isl.LTX_ARRAY_LINEEND,
        Rf"\(\cdot\) Data for period, {_data_periods[1].replace('-', '--')}",
        "calculated by subtracting reported figures for 1996--{}".format(
            int(_data_periods[1].split("-")[0]) - 1
        ),
        "from reported figures for 1996--{}".format(_data_periods[1].split("-")[1]),
    ))
    _stats_group_dict = {
        isl.StatsGrpSelector.FC: {
            "desc": f"{_test_regime.capitalize()} rates by Firm Count",
            "title_str": "By Number of Significant Competitors",
            "hval": "Firm Count",
            "hcol_width": 54,
            "notewidth": 0.67,
            "notestr": " ".join((
                _notes_str_base,
                isl.LTX_ARRAY_LINEEND,
                isl.LTX_ARRAY_LINEEND,
                isl.LTX_ARRAY_LINEEND,
            )),
        },
        isl.StatsGrpSelector.DL: {
            "desc": Rf"{_test_regime.capitalize()} rates by range of \(\Delta HHI\)",
            "title_str": R"By Change in Concentration (\Deltah{})",
            "hval": R"$\Delta HHI$",
            "hval_plus": R"{ $[\Delta_L, \Delta_H)$ pts.}",
            "hcol_width": 54,
            "notewidth": 0.67,
            "notestr": " ".join((
                _notes_str_base,
                isl.LTX_ARRAY_LINEEND,
                isl.LTX_ARRAY_LINEEND,
            )),
            "notestr_plus": " ".join((
                R"\(\cdot\) Ranges of $\Delta HHI$ are defined as",
                "half-open intervals with",
                R"$\Delta_L \leqslant \Delta HHI < \Delta_H$, except that",
                R"$2500 \text{ pts.} \leqslant \Delta HHI \leqslant 5000 \text{ pts.}$",
                R"in the closed interval [2500, 5000] pts.",
                isl.LTX_ARRAY_LINEEND,
                isl.LTX_ARRAY_LINEEND,
            )),
        },
        isl.StatsGrpSelector.ZN: {
            "desc": f"{_test_regime.capitalize()} rates by Approximate Presumption Zone",
            "title_str": R"By Approximate \textit{2010 Guidelines} Concentration-Based Standards",
            "hval": "Approximate Standard",
            "hcol_width": 190,
            "notewidth": 0.96,
            "notestr": " ".join((
                _notes_str_base,
                isl.LTX_ARRAY_LINEEND,
                isl.LTX_ARRAY_LINEEND,
            )),
        },
    }

    _output_dottex_pathlist = ()
    for _stats_group_key in _stats_group_dict:
        _output_dottex_path = _invres_stats_obs_render(
            _data_array_dict,
            _data_periods,
            _merger_classes,
            _stats_group_key,
            _stats_group_dict[_stats_group_key],
            _test_regime,
        )
        _output_dottex_pathlist += (_output_dottex_path,)

    return _output_dottex_pathlist


def _invres_stats_obs_render(
    _data_array_dict: Mapping,
    _data_periods: tuple[str, str],
    _merger_classes: Sequence[isl.INDGRPConstants | isl.EVIDENConstants],
    _stats_group: isl.StatsGrpSelector,
    _stats_group_dict: Mapping,
    _test_regime: isl.PolicySelector = isl.PolicySelector.CLRN,
    /,
) -> str:
    _invres_rate_table_content = isl.StatsContainer()
    _invres_rate_table_design = isl.latex_jinja_env.get_template(
        "ftcinvdata_summarypaired_table_template.tex.jinja2"
    )

    print(
        f'{_stats_group_dict["desc"]}:', ", ".join([f'"{g}"' for g in _merger_classes])
    )
    _invres_rate_table_content.test_regime = _test_regime.capitalize()
    _invres_rate_table_content.obs_summary_type = f"{_stats_group}"
    _invres_rate_table_content.obs_summary_type_title = _stats_group_dict.get(
        "title_str"
    )
    _invres_rate_table_content.hdrcol_raw_width = f'{_stats_group_dict["hcol_width"]}pt'

    _hs1 = _stats_group_dict["hval"]
    _hs2 = _h if (_h := _stats_group_dict.get("hval_plus", "")) else _hs1
    _invres_rate_table_content.invdata_hdrcoldescstr = (
        isl.latex_hrdcoldesc_format_str.format(
            "hdrcol_raw",
            f'{_stats_group_dict["hcol_width"]}pt',
            "hdrcoldesc_raw",
            "center",
            " ".join((
                _hs1 if _hs1 != _hs2 else Rf"{{ \phantom{{{_hs1}}} }}",
                isl.LTX_ARRAY_LINEEND,
                _hs2,
                isl.LTX_ARRAY_LINEEND,
            )),
        )
    )
    del _hs1, _hs2

    _invres_rate_table_content.obs_merger_class_0 = f"{_merger_classes[0]}"
    _invres_rate_table_content.obs_merger_class_1 = f"{_merger_classes[1]}"
    _invres_rate_table_content.obs_periods_str = (
        Rf"{' & '.join(_data_periods)} \\".replace("-", "--")
    )

    _invres_rate_table_content.invdata_notewidth = _stats_group_dict["notewidth"]
    _invres_rate_table_content.invdata_notestr = _stats_group_dict["notestr"]
    if _n2 := _stats_group_dict.get("notestr_plus", ""):
        _invres_rate_table_content.invdata_notestr += _n2
    del _n2

    _invdata_sourcestr_format_str = "{} {}".format(
        R"\(\cdot\) Fed. Trade Comm'n ({}), at note~\cref{{fn:{}}}",
        isl.LTX_ARRAY_LINEEND,
    )

    _table_nos = get_table_nos(
        _data_array_dict, _merger_classes, _stats_group, _data_periods[0]
    )
    _src_table_nos_str = (
        f'{", ".join(_table_nos[:-1])} and {_table_nos[-1]}'
        if _merger_classes[0] != isl.EVIDENConstants.NE
        else "Table~{0}.1, Table~{1}.1, and Table~{1}.2".format(
            *(4, 10) if _stats_group == "ByFirmCount" else (3, 9)
        )
    )
    _invres_rate_table_content.invdata_sourcestr = _invdata_sourcestr_format_str.format(
        "2003", "FTCInvData1996to2003"
    )
    _invres_rate_table_content.invdata_sourcestr += (
        _invdata_sourcestr_format_str.format("2011", "FTCInvData1996to2011")
    )

    _invdata_hdr_list: list[str] = []
    _invdata_dat_list: list[list[str]] = []
    _invres_cnt_totals: list[str] = []
    _sort_order = (
        isl.SortSelector.UCH
        if _stats_group == isl.StatsGrpSelector.FC
        else isl.SortSelector.REV
    )

    for _merger_class in _merger_classes:
        _table_ind_group = (
            _merger_class
            if isinstance(_merger_class, isl.INDGRPConstants)
            else isl.INDGRPConstants.ALL
        )
        _table_evid_cond = (
            _merger_class
            if isinstance(_merger_class, isl.EVIDENConstants)
            else isl.EVIDENConstants.UR
        )
        for _data_period in _data_periods:
            _invres_cnt_totals += [
                isl.invres_stats_output(
                    _data_array_dict,
                    _data_period,
                    _table_ind_group,
                    _table_evid_cond,
                    _stats_group,
                    _test_regime,
                    return_type_sel=isl.StatsReturnSelector.CNT,
                    print_to_screen=False,
                )[1][-1][0]
            ]

            _invdata_hdr_list_it, _invdata_dat_list_it = isl.invres_stats_output(
                _data_array_dict,
                _data_period,
                _table_ind_group,
                _table_evid_cond,
                _stats_group,
                _test_regime,
                return_type_sel=isl.StatsReturnSelector.RPT,
                sort_order=_sort_order,
                print_to_screen=False,
            )
            _invdata_hdr_list = _invdata_hdr_list_it
            _invdata_dat_list = (
                _invdata_dat_list_it[:]
                if not _invdata_dat_list
                else [
                    _invdata_dat_list[_r][:] + _invdata_dat_list_it[_r][:]
                    for _r in range(len(_invdata_dat_list))
                ]
            )

    isl.stats_print_rows(_invdata_hdr_list, _invdata_dat_list)

    _invdata_hdrstr = "".join([
        f"{_invdata_hdr_list[g]} {isl.LTX_ARRAY_LINEEND}"
        for g in range(len(_invdata_hdr_list))
    ])

    _invdata_datstr = "".join([
        f"{' & '.join(_invdata_dat_list[g])} {isl.LTX_ARRAY_LINEEND}"
        for g in range(len(_invdata_dat_list))
    ])

    (
        _invres_rate_table_content.mkt_counts_str_class_0,
        _invres_rate_table_content.mkt_counts_str_class_1,
    ) = (
        R"{} \\".format(" & ".join([f"Obs. = {f}" for f in g]))
        for g in [
            _invres_cnt_totals[: len(_data_periods)],
            _invres_cnt_totals[len(_data_periods) :],
        ]
    )

    _invres_rate_table_content.invdata_numrows = len(_invdata_hdr_list)
    _invres_rate_table_content.invdata_hdrstr = _invdata_hdrstr
    _invres_rate_table_content.invdata_datstr = _invdata_datstr

    _output_dottex_path = DATA_DIR / INVDATA_DOTTEX_FORMAT_STR.format(_stats_group)
    with _output_dottex_path.open("w", encoding="UTF-8") as _output_dottex_file:
        _output_dottex_file.write(
            _invres_rate_table_design.render(tmpl_data=_invres_rate_table_content)
        )
        print("\n", file=_output_dottex_file)
    del _invdata_hdrstr, _invdata_datstr

    return _output_dottex_path.name


def get_table_nos(
    _data_array_dict: Mapping,
    _merger_classes: Sequence[isl.INDGRPConstants | isl.EVIDENConstants],
    _stats_group: isl.StatsGrpSelector,
    _data_period: str,
    /,
) -> list:
    _stats_group_major = (
        "ByFirmCount" if _stats_group == isl.StatsGrpSelector.FC else "ByHHIandDelta"
    )

    _table_ind_groups = tuple(
        (_m if isinstance(_m, isl.INDGRPConstants) else isl.INDGRPConstants.ALL)
        for _m in _merger_classes
    )
    _table_evid_conds = tuple(
        (_m if isinstance(_m, isl.EVIDENConstants) else isl.EVIDENConstants.UR)
        for _m in _merger_classes
    )

    return list(
        dict.fromkeys(
            isl.table_no_lku(
                _data_array_dict[_data_period][_stats_group_major],
                _table_ind_group,
                _table_evid_cond,
            )
            for _table_ind_group in _table_ind_groups
            for _table_evid_cond in _table_evid_conds
        )
    )


if __name__ == "__main__":
    invdata_array_dict = fid.construct_data(
        fid.INVDATA_ARCHIVE_PATH,
        flag_backward_compatibility=False,
        flag_pharma_for_exclusion=True,
    )

    merger_classes = (
        isl.EVIDENConstants.NE,
        isl.EVIDENConstants.ED,
    )  # clstl.INDGRPConstants.IID)
    data_periods = ("1996-2003", "2004-2011")
    test_regime = isl.PolicySelector.ENFT

    # Now generate the various tables summarizing merger investigations data
    invres_stats_byhhianddelta_pathlist = invres_stats_odds_ratio_byhhianddelta(
        invdata_array_dict, data_periods, merger_classes
    )
    print(invres_stats_byhhianddelta_pathlist)
    invres_stats_allothers_pathlist = invres_stats_obs_setup(
        invdata_array_dict, data_periods, merger_classes, test_regime
    )
    isl.render_table_pdf(
        invres_stats_byhhianddelta_pathlist,
        INVDATA_DOTTEX_FORMAT_STR.format("ByHHIandDelta"),
    )
    isl.render_table_pdf(
        invres_stats_allothers_pathlist, INVDATA_DOTTEX_FORMAT_STR.format("AllOthers")
    )
