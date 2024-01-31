"""
Analyze simulated data on merger enforcement under U.S. Horizontal Merger Guidelines.

Format output as LaTeX tables (using TikZ).

"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Literal

import numpy as np
import re2 as re  # type: ignore
import tables as ptb  # type: ignore

import mergeron.gen.data_generation as dgl
import mergeron.gen.guidelines_tests as gsftl
import mergeron.gen.investigations_stats as clstl
from mergeron.core.gsf_lib import GuidelinesStandards
from mergeron.core.parse_ftc_merger_investigations_data import (
    construct_invdata,
    invdata_dump_path,
)
from mergeron.core.propn_ci_lib import propn_ci

__version__ = metadata.version(Path(__file__).parents[1].stem)

mod_path = Path(__file__)
data_path = Path.home() / mod_path.parents[1].stem

dottex_format_str = (
    "BenchmarkingGUPPISafeharbor_DRAFT_FTCClearanceRateCITables_{}_{}_SYM.tex"
)
clrenf_rate_table_content = clstl.ClearanceRateDataContainer()
clrenf_rate_table_design = clstl.latex_jinja_env.get_template(
    "clrrate_cis_summary_table_template.tex.jinja2"
)


def clrenf_stats_sim_setup(
    _data_array_dict: Mapping,
    _data_period: str,
    _merger_class: clstl.INDGRPConstants | clstl.EVIDENConstants,
    _clrenf_parm_vec: Sequence,
    _ind_sample_spec: dgl.MarketSampleSpec,
    _clr_enf_stats_kwargs: Mapping | None = None,
    /,
) -> None:
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

    _sim_clrenf_sel = _clr_enf_stats_kwargs["sim_clrenf_sel"]
    _clrenf_sel, _guppi_wgtng_policy, _divr_wgtng_policy = _sim_clrenf_sel

    # Get observed rates
    (
        _clrenf_cnts_obs_byfirmcount_array,
        _clrenf_cnts_obs_bydelta_array,
        _clrenf_cnts_obs_byconczone_array,
    ) = (
        clstl.clrenf_stats_cnts_by_group(
            _data_array_dict,
            _data_period,
            _table_ind_group,
            _table_evid_cond,
            _grp,
            _clrenf_sel,
        )
        for _grp in (
            clstl.StatsGrpSelector.FC,
            clstl.StatsGrpSelector.DL,
            clstl.StatsGrpSelector.ZN,
        )
    )

    _ind_sample_spec_here = deepcopy(_ind_sample_spec)
    _ind_sample_spec_here.share_spec = (
        dgl.SHRConstants.DIR_FLAT,
        dgl.RECConstants.INOUT,
        _clrenf_cnts_obs_byfirmcount_array[:-1, 1],
    )

    # Generate simulated rates
    _start_time = datetime.now()
    (
        _clrenf_cnts_sim_byfirmcount_array,
        _clrenf_cnts_sim_bydelta_array,
        _clrenf_cnts_sim_byconczone_array,
    ) = gsftl.sim_clrenf_cnts_ll(
        _clrenf_parm_vec, _ind_sample_spec_here, _clr_enf_stats_kwargs
    )
    _total_duration = datetime.now() - _start_time

    print(
        f"estimations completed in {_total_duration / timedelta(seconds=1):.6f} secs."
    )

    # Prepare and write/print output tables
    _stats_group_dict = {
        clstl.StatsGrpSelector.FC: {
            "desc": f"{_clrenf_sel.capitalize()} rates by firm count",
            "title_str": "By Number of Significant Competitors",
            "hval": "Firm Count",
            "hcol_width": 54,
            "notewidth": 0.63,
            "obs_array": _clrenf_cnts_obs_byfirmcount_array,
            "sim_array": _clrenf_cnts_sim_byfirmcount_array,
        },
        clstl.StatsGrpSelector.DL: {
            "desc": Rf"{_clrenf_sel.capitalize()} rates by range of $\Delta HHI$",
            "title_str": R"By Change in Concentration (\Deltah{})",
            "hval": R"$\Delta HHI$",
            "hval_plus": R"{ $[\Delta_L, \Delta_H)$ }",
            "hcol_width": 54,
            "notewidth": 0.63,
            "notestr_plus": " ".join((
                R"\(\cdot\) Ranges of $\Delta HHI$ are defined as",
                "half-open intervals with",
                R"$\Delta_L \leqslant \Delta HHI < \Delta_H$, except that",
                R"$2500 \text{ pts.} \leqslant \Delta HHI \leqslant 5000 \text{ pts.}$",
                R"in the closed interval [2500, 5000]",
                clstl.ltx_array_lineend,
                clstl.ltx_array_lineend,
            )),
            "obs_array": _clrenf_cnts_obs_bydelta_array,
            "sim_array": _clrenf_cnts_sim_bydelta_array,
        },
        clstl.StatsGrpSelector.ZN: {
            "desc": f"{_clrenf_sel.capitalize()} rates by Approximate Presumption Zone",
            "title_str": "{} {}".format(
                R"By Approximate \textit{2010 Guidelines}",
                "Concentration-Based Standards",
            ),
            "hval": "Approximate Standard",
            "hcol_width": 190,
            "notewidth": 0.96,
            "obs_array": _clrenf_cnts_obs_byconczone_array,
            "sim_array": _clrenf_cnts_sim_byconczone_array,
        },
    }

    with (
        data_path
        / dottex_format_str.format(
            _merger_class.replace(" ", ""), _data_period.split("-")[1]
        )
    ).open("w", encoding="UTF-8") as _clrenf_rate_table_dottex:
        for _stats_group_key in _stats_group_dict:
            clrenf_stats_sim_render(
                _data_period,
                _merger_class,
                _stats_group_key,
                _stats_group_dict[_stats_group_key],
                _clrenf_parm_vec,
                _clr_enf_stats_kwargs["sim_clrenf_sel"],
                _clrenf_rate_table_dottex,
            )


def clrenf_stats_sim_render(
    _data_period: str,
    _merger_class: clstl.INDGRPConstants | clstl.EVIDENConstants,
    _stats_group: clstl.StatsGrpSelector,
    _stats_group_dict_sub: Mapping,
    _clrenf_parm_vec: Sequence,
    _sim_clrenf_sel: Sequence,
    _clrenf_rate_table_dottex: Path.file,
    /,
) -> None:
    _clrenf_rate_table_content = clstl.ClearanceRateDataContainer()

    _obs_sample_sz, _sim_sample_sz = (
        np.einsum("i->", _stats_group_dict_sub[_g][:, _h]).astype(int)
        for _g, _h in (("obs_array", -2), ("sim_array", -5))
    )

    _clrenf_sel, _, _ = _sim_clrenf_sel
    (
        _clrenf_rate_table_content.clrenf_sel,
        _clrenf_rate_table_content.obs_merger_class,
        _clrenf_rate_table_content.obs_period,
    ) = (_clrenf_sel.capitalize(), _merger_class, _data_period.split("-"))

    _r_bar, _d_bar, _g_bar, _ipr_bar, _cmcr_bar = _clrenf_parm_vec
    (
        _clrenf_rate_table_content.rbar,
        _clrenf_rate_table_content.dbar,
        _clrenf_rate_table_content.guppi_bound,
        _clrenf_rate_table_content.ipr_bound,
        _clrenf_rate_table_content.cmcr_bound,
    ) = (rf"{_s * 100:.1f}\%" for _s in _clrenf_parm_vec)

    # Prepare and write/print output tables
    _clrenf_rate_cis_wilson_notestr = " ".join((
        R"Confidence intervals (95\% CI) are",
        R"estimated by the Wilson method, given the",
        "reported numbers of investigated mergers and cleared mergers",
        _stats_group_dict_sub["desc"].replace(
            f"{_clrenf_rate_table_content.clrenf_sel} rates ", ""
        ),
        clstl.ltx_array_lineend,
    ))

    _eg_count = (_relfreq_eg := 0.01) * _sim_sample_sz
    _clrenf_rate_sim_ci_eg = 100 * np.array(
        propn_ci(0.50 * _eg_count, _eg_count, method="Exact")
    )
    _clrenf_rate_sim_notestr = " ".join((
        R"\(\cdot\) Simulated {} rates are estimated by".format(
            _clrenf_rate_table_content.clrenf_sel
        ),
        "Monte Carlo integration over generated data representing",
        Rf"{_sim_sample_sz:,d} hypothetical mergers. Thus,",
        R"for a subset of simulations with a relative frequency",
        Rf"of, say, {100 * _relfreq_eg:.1f}\%,",
        R"and an estimated clearance rate of, for example, 50.0\%,",
        R"the margin of error (m.o.e.) is {}.".format(
            clstl.moe_tmpl.render(rv=_clrenf_rate_sim_ci_eg)
        ),
        R"The m.o.e. is derived from the",
        R"Clopper-Pearson exact 95\% confidence interval, {}.".format(
            clstl.ci_propn_format_str.format(*_clrenf_rate_sim_ci_eg).strip()
        ),
        R"(The m.o.e. for simulated clearance rates varies",
        R"as the reciprocal of the square root of the number",
        R"of hypothetical mergers.)",
        clstl.ltx_array_lineend,
    ))
    del _relfreq_eg, _eg_count, _clrenf_rate_sim_ci_eg

    print(
        f"Observed {_clrenf_sel} proportion [95% CI]",
        _stats_group_dict_sub["desc"].replace(f"{_clrenf_sel} rates ", ""),
    )
    print(f"\t with sample size (observed): {_obs_sample_sz:,d};")

    _clrenf_rate_table_content.obs_summary_type = f"{_stats_group}"
    _clrenf_rate_table_content.obs_summary_type_title = _stats_group_dict_sub["desc"]
    _clrenf_rate_table_content.clrenf_rate_cis_notewidth = _stats_group_dict_sub[
        "notewidth"
    ]
    _clrenf_rate_cis_numobs_notestr = " ".join((
        R"\(\cdot\) Estimates for Proportion {} are based on".format(
            "Enforced" if _clrenf_sel == clstl.CLRENFSelector.ENFT else "Cleared"
        ),
        f"{_obs_sample_sz:,d} total observations (investigated mergers).",
    ))

    _spnum = 2 if _stats_group_dict_sub["notewidth"] < 0.90 else 1
    _clrenf_rate_table_content.clrenf_rate_cis_notestr = " ".join((
        _clrenf_rate_cis_numobs_notestr,
        _clrenf_rate_cis_wilson_notestr,
        *[clstl.ltx_array_lineend] * _spnum,
        _clrenf_rate_sim_notestr,
        *[clstl.ltx_array_lineend] * (_spnum + 1),
    ))
    del _spnum

    if _nsp := _stats_group_dict_sub.get("notestr_plus", ""):
        _clrenf_rate_table_content.clrenf_rate_cis_notestr += "".join((
            clstl.ltx_array_lineend,
            _nsp,
        ))
    del _nsp

    _clrenf_stats_report_func = (
        clstl.latex_tbl_clrenf_stats_byzone
        if _stats_group == clstl.StatsGrpSelector.ZN
        else clstl.latex_tbl_clrenf_stats_1dim
    )
    _sort_order = (
        clstl.SortSelector.UCH
        if _stats_group == clstl.StatsGrpSelector.FC
        else clstl.SortSelector.REV
    )

    _clrenf_stats_hdr_list, _clrenf_stats_dat_list = _clrenf_stats_report_func(
        _stats_group_dict_sub["obs_array"],
        return_type_sel=clstl.StatsReturnSelector.RIN,
        sort_order=_sort_order,
    )
    if _stats_group == clstl.StatsGrpSelector.FC:
        del _clrenf_stats_hdr_list[-2], _clrenf_stats_dat_list[-2]

    _clrenf_rate_table_content.clrenf_rate_numrows = len(_clrenf_stats_hdr_list)
    _clrenf_rate_table_content.hdrcol_cis_width = (
        f'{_stats_group_dict_sub["hcol_width"]}pt'
    )

    _hs1 = _stats_group_dict_sub["hval"]
    _hs2 = _h if (_h := _stats_group_dict_sub.get("hval_plus", "")) else _hs1
    _clrenf_rate_table_content.hdrcoldescstr = clstl.format_latex_hrdcoldesc.format(
        "hdrcol_cis",
        f'{_stats_group_dict_sub["hcol_width"]}pt',
        "hdrcoldesc_cis",
        "center",
        " ".join((
            _hs1 if _hs1 != _hs2 else Rf"{{ \phantom{{{_hs1}}} }}",
            clstl.ltx_array_lineend,
            _hs2,
            clstl.ltx_array_lineend,
        )),
    )
    del _hs1, _hs2

    _clrenf_rate_table_content.clrenf_rate_hdrstr = "".join([
        f"{g} {clstl.ltx_array_lineend}" for g in _clrenf_stats_hdr_list
    ])
    _clrenf_rate_table_content.clrenf_rate_cis = "".join([
        f"{' & '.join(g)} {clstl.ltx_array_lineend}" for g in _clrenf_stats_dat_list
    ])
    print(_clrenf_rate_table_content.clrenf_rate_cis)
    print()
    del _clrenf_stats_hdr_list, _clrenf_stats_dat_list

    print(f"Simulated {_clrenf_sel} rates {_stats_group}:")
    print(f"\t with generated data size = {_sim_sample_sz:,d}:")

    _clrenf_rate_sim_hdr_list, _clrenf_rate_sim_dat_list = _clrenf_stats_report_func(
        _stats_group_dict_sub["sim_array"],
        return_type_sel=clstl.StatsReturnSelector.RPT,
        sort_order=_sort_order,
    )
    _clrenf_rate_table_content.clrenf_rate_sim = "".join([
        f"{' & '.join(g)} {clstl.ltx_array_lineend}" for g in _clrenf_rate_sim_dat_list
    ])
    del _clrenf_rate_sim_hdr_list, _clrenf_rate_sim_dat_list
    print(_clrenf_rate_table_content.clrenf_rate_sim)
    print()

    # Generate and write out LaTeX
    _clrenf_rate_table_design = clstl.latex_jinja_env.get_template(
        "clrrate_cis_summary_table_template.tex.jinja2"
    )
    # Write to dottex
    _clrenf_rate_table_dottex.write(
        _clrenf_rate_table_design.render(tmpl_data=_clrenf_rate_table_content)
    )
    print("\n", file=_clrenf_rate_table_dottex)


if __name__ == "__main__":
    invdata_array_dict = construct_invdata(
        invdata_dump_path,
        flag_backward_compatibility=False,
        flag_pharma_for_exclusion=True,
    )

    data_periods = ("1996-2003", "2004-2011")
    merger_classes = (clstl.INDGRPConstants.ALL, clstl.EVIDENConstants.ED)

    sample_sz_base = 10**6
    pr_sym_spec = dgl.PRIConstants.SYM
    pcm_dist_type, pcm_dist_parms = dgl.PCMConstants.EMPR, np.empty(2)

    save_data_to_file_flag = False
    save_data_to_file: Literal[False] | tuple[Literal[True], ptb.File, str] = False
    if save_data_to_file_flag:
        h5path = data_path.joinpath(f"{mod_path.stem}.h5")
        blosc_filters = ptb.Filters(complevel=3, complib="blosc:lz4", fletcher32=True)
        h5datafile = ptb.open_file(
            str(h5path),
            mode="w",
            title="Datasets, Sound GUPPI Safeharbor",
            filters=blosc_filters,
        )
        save_data_to_file = (True, h5datafile, "/")

    sim_clrenf_sel = (
        (clstl.CLRENFSelector.CLRN, gsftl.GUPPIWghtngSelector.MAX, None),
        (clstl.CLRENFSelector.ENFT, gsftl.GUPPIWghtngSelector.OSD, None),
    )[1]
    clrenf_stats_kwargs = {"sim_clrenf_sel": sim_clrenf_sel}

    for merger_class in merger_classes:
        for study_period in data_periods:
            if study_period == data_periods[1]:
                continue

            print(
                f"{sim_clrenf_sel[0].capitalize()} rates and c.i.s",
                f"for the class of mergers, '{merger_class}',",
                f"for study period, {study_period}:",
            )
            clrenf_rate_table_content.obs_period = study_period.split("-")

            clrenf_parm_vec = (
                GuidelinesStandards(2010).presumption[2:]
                if study_period.split("-")[1] == data_periods[1].split("-")[1]
                else GuidelinesStandards(1992).presumption[2:]
            )

            ind_sample_spec = dgl.MarketSampleSpec(
                sample_sz_base,
                clrenf_parm_vec[0],
                pr_sym_spec,
                pcm_spec=(pcm_dist_type, dgl.FM2Constants.MNL, pcm_dist_parms),
                hsr_filing_test_type=dgl.SSZConstants.HSR_NTH,
            )

            # A file to write tables to, and a hierarchy under which to store the data
            if save_data_to_file:
                h5hier_pat = re.compile(r"\W")
                h5hier = f"/{h5hier_pat.sub('_', f'{merger_class} {study_period}')}"
                save_data_to_file = save_data_to_file[:-1] + (h5hier,)

            clrenf_stats_sim_setup(
                invdata_array_dict,
                study_period,
                merger_class,
                clrenf_parm_vec,
                ind_sample_spec,
                clrenf_stats_kwargs,
            )

    if save_data_to_file:
        h5datafile.close()
