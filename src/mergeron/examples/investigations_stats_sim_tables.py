"""
Analyze simulated data on merger enforcement under U.S. Horizontal Merger Guidelines.

Format output as LaTeX tables (using TikZ).

"""

import sys
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Literal

import numpy as np
import re2 as re  # type: ignore
import tables as ptb  # type: ignore
from attrs import evolve

import mergeron.core.ftc_merger_investigations_data as fid
import mergeron.gen.data_generation as dgl
import mergeron.gen.guidelines_tests as gtl
import mergeron.gen.investigations_stats as isl
from mergeron import DATA_DIR
from mergeron.core.guidelines_standards import GuidelinesStandards
from mergeron.core.proportions_tests import propn_ci

if not sys.warnoptions:
    warnings.simplefilter("ignore")  # , category="RuntimeWarning")

dottex_format_str = "BenchmarkingGUPPISafeharbor_DRAFT_FTC{}RateCITables_{}_{}_SYM.tex"
stats_table_content = isl.StatsContainer()
stats_table_design = isl.latex_jinja_env.get_template(
    "clrrate_cis_summary_table_template.tex.jinja2"
)


def invres_stats_sim_setup(
    _invdata: fid.INVData,
    _data_period: str,
    _merger_class: isl.INDGRPConstants | isl.EVIDENConstants,
    _invres_parm_vec: Sequence[float],
    _sample_spec: dgl.MarketSampleSpec,
    _invres_stats_kwargs: Mapping[str, Any] | None = None,
    /,
) -> None:
    _table_ind_group = (
        _merger_class
        if _merger_class == isl.INDGRPConstants.IIC
        else isl.INDGRPConstants.ALL
    )
    _table_evid_cond = (
        _merger_class
        if isinstance(_merger_class, isl.EVIDENConstants)
        else isl.EVIDENConstants.UR
    )

    _invres_stats_kwargs = _invres_stats_kwargs or {
        "sim_test_regime": isl.PolicySelector.ENFT
    }
    _sim_test_regime = _invres_stats_kwargs["sim_test_regime"]
    _test_regime, _guppi_wgtng_policy, _divr_wgtng_policy = _sim_test_regime

    # Get observed rates
    (
        _invres_cnts_obs_byfirmcount_array,
        _invres_cnts_obs_bydelta_array,
        _invres_cnts_obs_byconczone_array,
    ) = (
        isl.invres_stats_cnts_by_group(
            _invdata,
            _data_period,
            _table_ind_group,
            _table_evid_cond,
            _grp,
            _test_regime,
        )
        for _grp in (
            isl.StatsGrpSelector.FC,
            isl.StatsGrpSelector.DL,
            isl.StatsGrpSelector.ZN,
        )
    )

    _sample_spec_here = evolve(
        _sample_spec,
        share_spec=dgl.ShareSpec(
            dgl.RECConstants.INOUT,
            dgl.SHRConstants.DIR_FLAT,
            None,
            _invres_cnts_obs_byfirmcount_array[:-1, 1],
        ),
    )

    # Generate simulated rates
    _start_time = datetime.now()
    (
        _invres_cnts_sim_byfirmcount_array,
        _invres_cnts_sim_bydelta_array,
        _invres_cnts_sim_byconczone_array,
    ) = gtl.sim_invres_cnts_ll(_invres_parm_vec, _sample_spec_here, _invres_stats_kwargs)
    _total_duration = datetime.now() - _start_time

    print(
        f"estimations completed in {_total_duration / timedelta(seconds=1):.6f} secs."
    )

    # Prepare and write/print output tables
    _stats_group_dict = {
        isl.StatsGrpSelector.FC: {
            "desc": f"{_test_regime.capitalize()} rates by firm count",
            "title_str": "By Number of Significant Competitors",
            "hval": "Firm Count",
            "hcol_width": 54,
            "notewidth": 0.63,
            "obs_array": _invres_cnts_obs_byfirmcount_array,
            "sim_array": _invres_cnts_sim_byfirmcount_array,
        },
        isl.StatsGrpSelector.DL: {
            "desc": Rf"{_test_regime.capitalize()} rates by range of $\Delta HHI$",
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
                isl.LTX_ARRAY_LINEEND,
                isl.LTX_ARRAY_LINEEND,
            )),
            "obs_array": _invres_cnts_obs_bydelta_array,
            "sim_array": _invres_cnts_sim_bydelta_array,
        },
        isl.StatsGrpSelector.ZN: {
            "desc": f"{_test_regime.capitalize()} rates by Approximate Presumption Zone",
            "title_str": "{} {}".format(
                R"By Approximate \textit{2010 Guidelines}",
                "Concentration-Based Standards",
            ),
            "hval": "Approximate Standard",
            "hcol_width": 190,
            "notewidth": 0.96,
            "obs_array": _invres_cnts_obs_byconczone_array,
            "sim_array": _invres_cnts_sim_byconczone_array,
        },
    }

    with (
        DATA_DIR
        / dottex_format_str.format(
            _test_regime.capitalize(),
            _merger_class.replace(" ", ""),
            _data_period.split("-")[1],
        )
    ).open("w", encoding="UTF-8") as _stats_table_dottex:
        for _stats_group_key in _stats_group_dict:
            invres_stats_sim_render(
                _data_period,
                _merger_class,
                _stats_group_key,
                _stats_group_dict[_stats_group_key],
                _invres_parm_vec,
                _invres_stats_kwargs["sim_test_regime"],
                _stats_table_dottex,
            )


def invres_stats_sim_render(
    _data_period: str,
    _merger_class: isl.INDGRPConstants | isl.EVIDENConstants,
    _stats_group: isl.StatsGrpSelector,
    _stats_group_dict_sub: Mapping[str, Any],
    _invres_parm_vec: Sequence[float],
    _sim_test_regime: gtl.UPPTestRegime,
    _stats_table_dottex: TextIOWrapper,
    /,
) -> None:
    _stats_table_content = isl.StatsContainer()

    _obs_sample_sz, _sim_sample_sz = (
        np.einsum("i->", _stats_group_dict_sub[_g][:, _h]).astype(int)
        for _g, _h in (("obs_array", -2), ("sim_array", -5))
    )

    _test_regime, _, _ = _sim_test_regime
    (
        _stats_table_content.test_regime,
        _stats_table_content.obs_merger_class,
        _stats_table_content.obs_period,
    ) = (_test_regime.capitalize(), _merger_class, _data_period.split("-"))

    _r_bar, _d_bar, _g_bar, _cmcr_bar, _ipr_bar = _invres_parm_vec
    (
        _stats_table_content.rbar,
        _stats_table_content.dbar,
        _stats_table_content.guppi_bound,
        _stats_table_content.ipr_bound,
        _stats_table_content.cmcr_bound,
    ) = (rf"{_s * 100:.1f}\%" for _s in _invres_parm_vec)

    # Prepare and write/print output tables
    _stats_cis_wilson_notestr = " ".join((
        R"Confidence intervals (95\% CI) are",
        R"estimated by the Wilson method, given the",
        "reported numbers of investigated mergers and cleared mergers",
        _stats_group_dict_sub["desc"].replace(
            f"{_stats_table_content.test_regime} rates ", ""
        ),
        isl.LTX_ARRAY_LINEEND,
    ))

    _eg_count = (_relfreq_eg := 0.01) * _sim_sample_sz
    _stats_sim_ci_eg = 100 * np.array(
        propn_ci(0.50 * _eg_count, _eg_count, method="Exact")
    )
    _stats_sim_notestr = " ".join((
        Rf"\(\cdot\) Simulated {_stats_table_content.test_regime} rates are estimated by",
        "Monte Carlo integration over generated data representing",
        Rf"{_sim_sample_sz:,d} hypothetical mergers. Thus,",
        R"for a subset of simulations with a relative frequency",
        Rf"of, say, {100 * _relfreq_eg:.1f}\%,",
        R"and an estimated clearance rate of, for example, 50.0\%,",
        Rf"the margin of error (m.o.e.) is {isl.moe_tmpl.render(rv=_stats_sim_ci_eg)}."
        R"The m.o.e. is derived from the",
        R"Clopper-Pearson exact 95\% confidence interval, {}.".format(
            isl.ci_format_str.format(*_stats_sim_ci_eg).strip()
        ),
        R"(The m.o.e. for simulated clearance rates varies",
        R"as the reciprocal of the square root of the number",
        R"of hypothetical mergers.)",
        isl.LTX_ARRAY_LINEEND,
    ))
    del _relfreq_eg, _eg_count, _stats_sim_ci_eg

    print(
        f"Observed {_test_regime} proportion [95% CI]",
        _stats_group_dict_sub["desc"].replace(f"{_test_regime} rates ", ""),
    )
    print(f"\t with sample size (observed): {_obs_sample_sz:,d};")

    _stats_table_content.obs_summary_type = f"{_stats_group}"
    _stats_table_content.obs_summary_type_title = _stats_group_dict_sub["desc"]
    _stats_table_content.stats_cis_notewidth = _stats_group_dict_sub["notewidth"]
    _stats_cis_numobs_notestr = " ".join((
        R"\(\cdot\) Estimates for Proportion {} are based on".format(
            "Enforced" if _test_regime == isl.PolicySelector.ENFT else "Cleared"
        ),
        f"{_obs_sample_sz:,d} total observations (investigated mergers).",
    ))

    _spnum = 2 if _stats_group_dict_sub["notewidth"] < 0.90 else 1
    _stats_table_content.stats_cis_notestr = " ".join((
        _stats_cis_numobs_notestr,
        _stats_cis_wilson_notestr,
        *[isl.LTX_ARRAY_LINEEND] * _spnum,
        _stats_sim_notestr,
        *[isl.LTX_ARRAY_LINEEND] * (_spnum + 1),
    ))
    del _spnum

    if _nsp := _stats_group_dict_sub.get("notestr_plus", ""):
        _stats_table_content.stats_cis_notestr += "".join((isl.LTX_ARRAY_LINEEND, _nsp))
    del _nsp

    _invres_stats_report_func = (
        isl.latex_tbl_invres_stats_byzone
        if _stats_group == isl.StatsGrpSelector.ZN
        else isl.latex_tbl_invres_stats_1dim
    )
    _sort_order = (
        isl.SortSelector.UCH
        if _stats_group == isl.StatsGrpSelector.FC
        else isl.SortSelector.REV
    )

    _invres_stats_hdr_list, _invres_stats_dat_list = _invres_stats_report_func(
        _stats_group_dict_sub["obs_array"],
        return_type_sel=isl.StatsReturnSelector.RIN,
        sort_order=_sort_order,
    )
    if _stats_group == isl.StatsGrpSelector.FC:
        del _invres_stats_hdr_list[-2], _invres_stats_dat_list[-2]

    _stats_table_content.stats_numrows = len(_invres_stats_hdr_list)
    _stats_table_content.hdrcol_cis_width = f'{_stats_group_dict_sub["hcol_width"]}pt'

    _hs1 = _stats_group_dict_sub["hval"]
    _hs2 = _h if (_h := _stats_group_dict_sub.get("hval_plus", "")) else _hs1
    _stats_table_content.hdrcoldescstr = isl.latex_hrdcoldesc_format_str.format(
        "hdrcol_cis",
        f'{_stats_group_dict_sub["hcol_width"]}pt',
        "hdrcoldesc_cis",
        "center",
        " ".join((
            _hs1 if _hs1 != _hs2 else Rf"{{ \phantom{{{_hs1}}} }}",
            isl.LTX_ARRAY_LINEEND,
            _hs2,
            isl.LTX_ARRAY_LINEEND,
        )),
    )
    del _hs1, _hs2

    _stats_table_content.stats_hdrstr = "".join([
        f"{g} {isl.LTX_ARRAY_LINEEND}" for g in _invres_stats_hdr_list
    ])
    _stats_table_content.stats_cis = "".join([
        f"{' & '.join(g)} {isl.LTX_ARRAY_LINEEND}" for g in _invres_stats_dat_list
    ])
    # Print to console
    isl.stats_print_rows(_invres_stats_hdr_list, _invres_stats_dat_list)
    del _invres_stats_hdr_list, _invres_stats_dat_list

    print(f"Simulated {_test_regime} rates {_stats_group}:")
    print(f"\t with generated data size = {_sim_sample_sz:,d}:")

    _stats_sim_hdr_list, _stats_sim_dat_list = _invres_stats_report_func(
        _stats_group_dict_sub["sim_array"],
        return_type_sel=isl.StatsReturnSelector.RPT,
        sort_order=_sort_order,
    )
    _stats_table_content.stats_sim = "".join([
        f"{' & '.join(g)} {isl.LTX_ARRAY_LINEEND}" for g in _stats_sim_dat_list
    ])
    # Print to console
    isl.stats_print_rows(_stats_sim_hdr_list, _stats_sim_dat_list)
    del _stats_sim_hdr_list, _stats_sim_dat_list

    # Generate and write out LaTeX
    _stats_table_design = isl.latex_jinja_env.get_template(
        "clrrate_cis_summary_table_template.tex.jinja2"
    )
    # Write to dottex
    _stats_table_dottex.write(
        _stats_table_design.render(tmpl_data=_stats_table_content)
    )
    print("\n", file=_stats_table_dottex)


if __name__ == "__main__":
    invdata_array_dict = fid.construct_data(
        fid.INVDATA_ARCHIVE_PATH,
        flag_backward_compatibility=False,
        flag_pharma_for_exclusion=True,
    )

    data_periods = ("1996-2003", "2004-2011")
    merger_classes: Sequence[isl.INDGRPConstants | isl.EVIDENConstants] = (
        isl.INDGRPConstants.ALL,
        isl.EVIDENConstants.ED,
    )

    sample_sz_base = 10**7
    pr_sym_spec = dgl.PRIConstants.SYM
    pcm_dist_type, pcm_dist_parms = dgl.PCMConstants.EMPR, np.empty(2)

    save_data_to_file_flag = False
    save_data_to_file: Literal[False] | tuple[Literal[True], ptb.File, str] = False
    if save_data_to_file_flag:
        h5path = DATA_DIR.joinpath(f"{Path(__file__).stem}.h5")
        blosc_filters = ptb.Filters(complevel=3, complib="blosc:lz4", fletcher32=True)
        h5datafile = ptb.open_file(
            str(h5path),
            mode="w",
            title="Datasets, Sound GUPPI Safeharbor",
            filters=blosc_filters,
        )
        save_data_to_file = (True, h5datafile, "/")

    sim_test_regime = (
        (isl.PolicySelector.CLRN, gtl.GUPPIWghtngSelector.MAX, None),
        (isl.PolicySelector.ENFT, gtl.GUPPIWghtngSelector.OSD, None),
    )[1]
    invres_stats_kwargs = {"sim_test_regime": sim_test_regime}

    for merger_class in merger_classes:
        for study_period in data_periods:
            if study_period == data_periods[1]:
                continue

            print(
                f"{sim_test_regime[0].capitalize()} rates and c.i.s",
                f"for the class of mergers, '{merger_class}',",
                f"for study period, {study_period}:",
            )
            stats_table_content.obs_period = study_period.split("-")

            clrenf_parm_vec = (
                GuidelinesStandards(2010).presumption[2:]
                if study_period.split("-")[1] == data_periods[1].split("-")[1]
                else GuidelinesStandards(1992).presumption[2:]
            )

            mkt_sample_spec = dgl.MarketSampleSpec(
                sample_sz_base,
                clrenf_parm_vec[0],
                pr_sym_spec,
                pcm_spec=dgl.PCMSpec(
                    pcm_dist_type, dgl.FM2Constants.MNL, pcm_dist_parms
                ),
                hsr_filing_test_type=dgl.SSZConstants.HSR_NTH,
            )

            # A file to write tables to, and a hierarchy under which to store the data
            if save_data_to_file:
                h5hier_pat = re.compile(r"\W")
                h5hier = f"/{h5hier_pat.sub('_', f'{merger_class} {study_period}')}"
                save_data_to_file = save_data_to_file[:-1] + (h5hier,)

            invres_stats_sim_setup(
                invdata_array_dict,
                study_period,
                merger_class,
                clrenf_parm_vec,
                mkt_sample_spec,
                invres_stats_kwargs,
            )

    if save_data_to_file:
        h5datafile.close()
