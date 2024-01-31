"""

    Construct observed clearance rates and enforcement rates, by specification,
    from FTC merger investigations data

"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from numpy import einsum, row_stack, unique

import mergeron.core.ftc_merger_investigations_data as fid
import mergeron.gen.investigations_stats as clstl

data_path = Path.home() / Path(__file__).parents[1].stem

clrenf_ratio_format_str = "{: >3.0f}/{:<3.0f}"
format_str_invdata_dottex_name = "{}.tex".format(
    "_".join((
        "BenchmarkingGUPPISafeharbor",
        "DRAFT",
        "FTCMergerInvestigationsDataTables",
        "{}",
        "OBS",
    ))
)


def clrenf_stats_odds_ratio_byhhianddelta(
    _data_array_dict: Mapping,
    _data_periods: tuple[str, str],
    _merger_classes: Sequence[clstl.INDGRPConstants | clstl.EVIDENConstants],
    /,
):
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
    _stats_group = clstl.StatsGrpSelector.HD
    _clrenf_rate_table_content = clstl.ClearanceRateDataContainer()
    _clrenf_rate_table_design = clstl.latex_jinja_env.get_template(
        "ftcinvdata_byhhianddelta_table_template.tex.jinja2"
    )
    _clrenf_rate_table_content.obs_summary_type = f"{_stats_group}"

    for _merger_class in _merger_classes:
        _table_ind_group = (
            _merger_class
            if _merger_class == clstl.INDGRPConstants.IIC
            else clstl.INDGRPConstants.ALL
        )
        _table_evid_cond = (
            _merger_class
            if isinstance(_merger_class, clstl.EVIDENConstants)
            else clstl.EVIDENConstants.UR
        )
        _clrenf_rate_table_content.obs_merger_class = f"{_merger_class}"

        for _data_period in _data_periods:
            _data_array_dict_sub = _data_array_dict[_data_period][f"{_stats_group}"]
            _table_no = clstl.table_no_lku(
                _data_array_dict_sub, _table_ind_group, _table_evid_cond
            )

            _clrenf_rate_table_content.table_ref = _table_no

            _data_period_0, _data_period_1 = (int(f) for f in _data_period.split("-"))
            if _data_period_0 != 1996:
                _clrenf_rate_table_content.invdata_notestr = " ".join((
                    "NOTES:",
                    clstl.ltx_array_lineend,
                    R"\(\cdot\) Data for period, {}".format(
                        _data_periods[1].replace("-", "--")
                    ),
                    f"calculated by subtracting reported figures for 1996--{_data_period_0 - 1}",
                    R"from reported figures for 1996--2011",
                    clstl.ltx_array_lineend,
                    clstl.ltx_array_lineend,
                ))

                _clrenf_rate_table_content.invdata_sourcestr = " ".join((
                    "\\(\\cdot\\) Fed. Trade Comm'n ({}), at~\\cref{{fn:{}}},".format(
                        _data_period_0, f"FTCInvData1996to{_data_period_0}"
                    ),
                    clstl.ltx_array_lineend,
                ))
                _clrenf_rate_table_content.invdata_sourcestr += " ".join((
                    "\\(\\cdot\\) Fed. Trade Comm'n ({}), at~\\cref{{fn:{}}},".format(
                        _data_period_1, f"FTCInvData1996to{_data_period_1}"
                    ),
                    clstl.ltx_array_lineend,
                ))
            else:
                _clrenf_rate_table_content.invdata_sourcestr = " ".join((
                    "\\(\\cdot\\) Fed. Trade Comm'n ({}), at~\\cref{{fn:{}}},".format(
                        _data_period_1, f"FTCInvData1996to{_data_period_1}"
                    ),
                    clstl.ltx_array_lineend,
                ))

            _clrenf_rate_table_content.obs_merger_class = f"{_merger_class}"
            _clrenf_rate_table_content.obs_period = _data_period.split("-")

            _clrenf_cnts_array = _data_array_dict_sub[_table_no][-1]
            _odds_ratio_data_str = ""
            for _hhi_range_it in unique(_clrenf_cnts_array[:, 0]):
                _clrenf_cnts_row_for_hhi_range = _clrenf_cnts_array[
                    _clrenf_cnts_array[:, 0] == _hhi_range_it
                ][:, 2:]
                _odds_ratio_data_str += " & ".join([
                    clrenf_ratio_format_str.format(*g)
                    for g in _clrenf_cnts_row_for_hhi_range
                ])
                _odds_ratio_data_str += " & {}".format(
                    clrenf_ratio_format_str.format(
                        *einsum("ij->j", _clrenf_cnts_row_for_hhi_range)
                    )
                )
                _odds_ratio_data_str += clstl.ltx_array_lineend

            _clrenf_cnts_row_for_hhi_tots = row_stack([
                einsum(
                    "ij->j", _clrenf_cnts_array[_clrenf_cnts_array[:, 1] == f][:, 2:]
                )
                for f in unique(_clrenf_cnts_array[:, 1])
            ])
            _odds_ratio_data_str += " & ".join([
                clrenf_ratio_format_str.format(*f)
                for f in _clrenf_cnts_row_for_hhi_tots
            ])
            _odds_ratio_data_str += " & {}".format(
                clrenf_ratio_format_str.format(
                    *einsum("ij->j", _clrenf_cnts_row_for_hhi_tots)
                )
            )
            _odds_ratio_data_str += clstl.ltx_array_lineend
            _clrenf_rate_table_content.invdata_byhhianddelta = _odds_ratio_data_str

            with (
                data_path
                / format_str_invdata_dottex_name.format(
                    f"{_stats_group}_{_data_period_1}_{_merger_class.replace(' ', '')}"
                )
            ).open("w", encoding="utf8") as _clrenf_rate_table_dottex:
                _clrenf_rate_table_dottex.write(
                    _clrenf_rate_table_design.render(
                        tmpl_data=_clrenf_rate_table_content
                    )
                )
                print("\n", file=_clrenf_rate_table_dottex)

            print(_odds_ratio_data_str)
            print()
            del _odds_ratio_data_str


def clrenf_stats_obs_setup(
    _data_array_dict: Mapping,
    _data_periods: tuple[str, str],
    _merger_classes: Sequence[clstl.INDGRPConstants | clstl.EVIDENConstants],
    _clrenf_sel: clstl.CLRENFSelector = clstl.CLRENFSelector.CLRN,
    /,
):
    _notes_str_base = " ".join((
        "NOTES:",
        clstl.ltx_array_lineend,
        Rf"\(\cdot\) Data for period, {_data_periods[1].replace('-', '--')}",
        "calculated by subtracting reported figures for 1996--{}".format(
            int(_data_periods[1].split("-")[0]) - 1
        ),
        "from reported figures for 1996--{}".format(_data_periods[1].split("-")[1]),
    ))
    _stats_group_dict = {
        clstl.StatsGrpSelector.FC: {
            "desc": f"{_clrenf_sel.capitalize()} rates by Firm Count",
            "title_str": "By Number of Significant Competitors",
            "hval": "Firm Count",
            "hcol_width": 54,
            "notewidth": 0.67,
            "notestr": " ".join((
                _notes_str_base,
                clstl.ltx_array_lineend,
                clstl.ltx_array_lineend,
                clstl.ltx_array_lineend,
            )),
        },
        clstl.StatsGrpSelector.DL: {
            "desc": Rf"{_clrenf_sel.capitalize()} rates by range of \(\Delta HHI\)",
            "title_str": R"By Change in Concentration (\Deltah{})",
            "hval": R"$\Delta HHI$",
            "hval_plus": R"{ $[\Delta_L, \Delta_H)$ pts.}",
            "hcol_width": 54,
            "notewidth": 0.67,
            "notestr": " ".join((
                _notes_str_base,
                clstl.ltx_array_lineend,
                clstl.ltx_array_lineend,
            )),
            "notestr_plus": " ".join((
                R"\(\cdot\) Ranges of $\Delta HHI$ are defined as",
                "half-open intervals with",
                R"$\Delta_L \leqslant \Delta HHI < \Delta_H$, except that",
                R"$2500 \text{ pts.} \leqslant \Delta HHI \leqslant 5000 \text{ pts.}$",
                R"in the closed interval [2500, 5000] pts.",
                clstl.ltx_array_lineend,
                clstl.ltx_array_lineend,
            )),
        },
        clstl.StatsGrpSelector.ZN: {
            "desc": f"{_clrenf_sel.capitalize()} rates by Approximate Presumption Zone",
            "title_str": R"By Approximate \textit{2010 Guidelines} Concentration-Based Standards",
            "hval": "Approximate Standard",
            "hcol_width": 190,
            "notewidth": 0.96,
            "notestr": " ".join((
                _notes_str_base,
                clstl.ltx_array_lineend,
                clstl.ltx_array_lineend,
            )),
        },
    }

    for _stats_group_key in _stats_group_dict:
        _clrenf_stats_obs_render(
            _data_array_dict,
            _data_periods,
            _merger_classes,
            _stats_group_key,
            _stats_group_dict[_stats_group_key],
            _clrenf_sel,
        )


def _clrenf_stats_obs_render(
    _data_array_dict: Mapping,
    _data_periods: tuple[str, str],
    _merger_classes: Sequence[clstl.INDGRPConstants | clstl.EVIDENConstants],
    _stats_group: clstl.StatsGrpSelector,
    _stats_group_dict: Mapping,
    _clrenf_sel: clstl.CLRENFSelector = clstl.CLRENFSelector.CLRN,
    /,
):
    _clrenf_rate_table_content = clstl.ClearanceRateDataContainer()
    _clrenf_rate_table_design = clstl.latex_jinja_env.get_template(
        "ftcinvdata_summarypaired_table_template.tex.jinja2"
    )

    print(
        f'{_stats_group_dict["desc"]}:', ", ".join([f'"{g}"' for g in _merger_classes])
    )
    _clrenf_rate_table_content.clrenf_sel = _clrenf_sel.capitalize()
    _clrenf_rate_table_content.obs_summary_type = f"{_stats_group}"
    _clrenf_rate_table_content.obs_summary_type_title = _stats_group_dict.get(
        "title_str"
    )
    _clrenf_rate_table_content.hdrcol_raw_width = f'{_stats_group_dict["hcol_width"]}pt'

    _hs1 = _stats_group_dict["hval"]
    _hs2 = _h if (_h := _stats_group_dict.get("hval_plus", "")) else _hs1
    _clrenf_rate_table_content.invdata_hdrcoldescstr = (
        clstl.format_latex_hrdcoldesc.format(
            "hdrcol_raw",
            f'{_stats_group_dict["hcol_width"]}pt',
            "hdrcoldesc_raw",
            "center",
            " ".join((
                _hs1 if _hs1 != _hs2 else Rf"{{ \phantom{{{_hs1}}} }}",
                clstl.ltx_array_lineend,
                _hs2,
                clstl.ltx_array_lineend,
            )),
        )
    )
    del _hs1, _hs2

    _clrenf_rate_table_content.obs_merger_class_0 = f"{_merger_classes[0]}"
    _clrenf_rate_table_content.obs_merger_class_1 = f"{_merger_classes[1]}"
    _clrenf_rate_table_content.obs_periods_str = (
        Rf"{' & '.join(_data_periods)} \\".replace("-", "--")
    )

    _clrenf_rate_table_content.invdata_notewidth = _stats_group_dict["notewidth"]
    _clrenf_rate_table_content.invdata_notestr = _stats_group_dict["notestr"]
    if _n2 := _stats_group_dict.get("notestr_plus", ""):
        _clrenf_rate_table_content.invdata_notestr += _n2
    del _n2

    _invdata_sourcestr_format_str = "{} {}".format(
        R"\(\cdot\) Fed. Trade Comm'n ({}), at note~\cref{{fn:{}}}",
        clstl.ltx_array_lineend,
    )

    _table_nos = get_table_nos(
        _data_array_dict, _merger_classes, _stats_group, _data_periods[0]
    )
    _src_table_nos_str = (
        f'{", ".join(_table_nos[:-1])} and {_table_nos[-1]}'
        if _merger_classes[0] != clstl.EVIDENConstants.NE
        else "Table~{0}.1, Table~{1}.1, and Table~{1}.2".format(
            *(4, 10) if _stats_group == "ByFirmCount" else (3, 9)
        )
    )
    _clrenf_rate_table_content.invdata_sourcestr = _invdata_sourcestr_format_str.format(
        "2003", "FTCInvData1996to2003"
    )
    _clrenf_rate_table_content.invdata_sourcestr += (
        _invdata_sourcestr_format_str.format("2011", "FTCInvData1996to2011")
    )

    _invdata_hdr_list: list[str] = []
    _invdata_dat_list: list[str] = []
    _clrenf_cnt_totals: list[str] = []
    _sort_order = (
        clstl.SortSelector.UCH
        if _stats_group == clstl.StatsGrpSelector.FC
        else clstl.SortSelector.REV
    )

    for _merger_class in _merger_classes:
        _table_ind_group = (
            _merger_class
            if _merger_class == clstl.INDGRPConstants.IIC
            else clstl.INDGRPConstants.ALL
        )
        _table_evid_cond = (
            _merger_class
            if isinstance(_merger_class, clstl.EVIDENConstants)
            else clstl.EVIDENConstants.UR
        )
        for _data_period in _data_periods:
            _clrenf_cnt_totals += [
                clstl.clrenf_stats_output(
                    _data_array_dict,
                    _data_period,
                    _table_ind_group,
                    _table_evid_cond,
                    _stats_group,
                    _clrenf_sel,
                    return_type_sel=clstl.StatsReturnSelector.CNT,
                    print_to_screen=False,
                )[1][-1][0]
            ]

            _invdata_hdr_list_it, _invdata_dat_list_it = clstl.clrenf_stats_output(
                _data_array_dict,
                _data_period,
                _table_ind_group,
                _table_evid_cond,
                _stats_group,
                _clrenf_sel,
                return_type_sel=clstl.StatsReturnSelector.RPT,
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

    _invdata_hdrstr = "".join([
        f"{_invdata_hdr_list[g]} {clstl.ltx_array_lineend}"
        for g in range(len(_invdata_hdr_list))
    ])

    _invdata_datstr = "".join([
        f'{" & ".join(_invdata_dat_list[g])} {clstl.ltx_array_lineend}'
        for g in range(len(_invdata_dat_list))
    ])

    for g in range(len(_invdata_hdr_list)):
        print(
            _invdata_hdr_list[g],
            " & ",
            " & ".join(_invdata_dat_list[g]),
            f" {clstl.ltx_array_lineend}",
            end="",
        )
    print()

    (
        _clrenf_rate_table_content.mkt_counts_str_class_0,
        _clrenf_rate_table_content.mkt_counts_str_class_1,
    ) = (
        R"{} \\".format(" & ".join([f"Obs. = {f}" for f in g]))
        for g in [
            _clrenf_cnt_totals[: len(_data_periods)],
            _clrenf_cnt_totals[len(_data_periods) :],
        ]
    )

    _clrenf_rate_table_content.invdata_numrows = len(_invdata_hdr_list)
    _clrenf_rate_table_content.invdata_hdrstr = _invdata_hdrstr
    _clrenf_rate_table_content.invdata_datstr = _invdata_datstr

    with (data_path / format_str_invdata_dottex_name.format(_stats_group)).open(
        "w", encoding="UTF-8"
    ) as _clrenf_rate_table_dottex:
        _clrenf_rate_table_dottex.write(
            _clrenf_rate_table_design.render(tmpl_data=_clrenf_rate_table_content)
        )
        print("\n", file=_clrenf_rate_table_dottex)
    del _invdata_hdrstr, _invdata_datstr


def get_table_nos(
    _data_array_dict: Mapping,
    _merger_classes: Sequence[clstl.INDGRPConstants | clstl.EVIDENConstants],
    _stats_group: clstl.StatsGrpSelector,
    _data_period: str,
    /,
) -> list:
    _stats_group_major = (
        "ByFirmCount" if _stats_group == clstl.StatsGrpSelector.FC else "ByHHIandDelta"
    )

    _table_ind_groups = tuple(
        (_m if _m == clstl.INDGRPConstants.IIC else clstl.INDGRPConstants.ALL)
        for _m in _merger_classes
    )
    _table_evid_conds = tuple(
        (_m if isinstance(_m, clstl.EVIDENConstants) else clstl.EVIDENConstants.UR)
        for _m in _merger_classes
    )

    return list(
        dict.fromkeys(
            clstl.table_no_lku(
                _data_array_dict[_data_period][_stats_group_major],
                _table_ind_group,
                _table_evid_cond,
            )
            for _table_ind_group in _table_ind_groups
            for _table_evid_cond in _table_evid_conds
        )
    )


if __name__ == "__main__":
    invdata_array_dict = fid.construct_invdata(
        fid.invdata_dump_path,
        flag_backward_compatibility=False,
        flag_pharma_for_exclusion=True,
    )

    merger_classes = (
        clstl.EVIDENConstants.NE,
        clstl.EVIDENConstants.ED,
    )  # clstl.INDGRPConstants.IID)
    data_periods = ("1996-2003", "2004-2011")
    clrenf_sel = clstl.CLRENFSelector.ENFT

    # Write the TiKZ setup file to destination first, for rendering the
    # .tex files of tables as PDF:
    with (
        data_path
        / "{}_{}_{}.tex".format(
            *format_str_invdata_dottex_name.split("_")[:2], "TikZTableSettings"
        )
    ).open("w", encoding="utf8") as _table_settings_dottex:
        _table_settings_dottex.write(
            clstl.latex_jinja_env.get_template("setup_tikz_tables.tex.jinja2").render()
        )
        print("\n", file=_table_settings_dottex)
    # Now generate the various tables summarizing merger investigations data
    clrenf_stats_odds_ratio_byhhianddelta(
        invdata_array_dict, data_periods, merger_classes
    )
    clrenf_stats_obs_setup(invdata_array_dict, data_periods, merger_classes, clrenf_sel)
