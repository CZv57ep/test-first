import re
from datetime import datetime, timedelta
from importlib import metadata
from itertools import product as iterprod
from pathlib import Path
from typing import Literal

import numpy as np
import tables as ptb  # type: ignore

import mergeron.core.guidelines_standards as gsf
import mergeron.gen.data_generation as dgl
import mergeron.gen.guidelines_tests as gtl
import mergeron.gen.investigations_stats as clstl

mod_name = Path(__file__).parents[1].stem
__version__ = metadata.version(mod_name)

mod_path = Path(__file__)
data_path = Path.home() / mod_name

tests_of_interest = (
    ("clearance", gtl.GUPPIWghtngSelector.MAX, gtl.GUPPIWghtngSelector.MAX),
    ("enforcement", gtl.GUPPIWghtngSelector.MIN, gtl.GUPPIWghtngSelector.MIN),
)


def analyze_clrenf_data(
    _sample_size: int = 10**6,
    _hmg_std_pub_year: Literal[1992, 2010, 2023] = 1992,
    _test_sel: tuple[
        str, gtl.GUPPIWghtngSelector, gtl.GUPPIWghtngSelector | None
    ] = tests_of_interest[1],
    /,
    *,
    save_data_to_file_flag: bool = False,
) -> None:
    """
    Analyze intrinsic enforcement rates using a GUPPI criterion against
    intrinsic enforcement rates by Guidelines ∆HHI standard

    Parameters
    ----------
    _sample_size
        Number of draws (mergers) to analyze

    _hmg_std_pub_year
        Guidelines version for ∆HHI standard

    _test_sel
        Specifies analysis of enforcement rates or, alternatively, clearance rates

    save_data_to_file_flag
        If True, simulated data are save to file (hdf5 format)

    """
    _clrenf_parm_vec = gsf.GuidelinesStandards(_hmg_std_pub_year).presumption[2:]

    _save_data_to_file: Literal[False] | tuple[Literal[True], ptb.File, str] = False
    if save_data_to_file_flag:
        _h5_hier_pat = re.compile(r"\W")
        _blosc_filters = ptb.Filters(
            complevel=3, complib="blosc:lz4hc", fletcher32=True
        )
        _h5_datafile = ptb.open_file(
            str(data_path / f"{mod_path.stem}_sound.h5"),
            mode="w",
            title=f"GUPPI Safeharbor {_test_sel[0].capitalize()} Rate Module",
            filters=_blosc_filters,
        )
        _h5_hier = f"/{_h5_hier_pat.sub('_', f'Standards from {_hmg_std_pub_year} Guidelines')}"

        _save_data_to_file = (True, _h5_datafile, _h5_hier)

    # ##
    #   Print summaries of intrinsic clearance/enforcement rates by ∆HHI,
    #   with asymmetric margins
    #  ##
    for recapture_spec_test, pcm_dist_test_tup, pcm_dist_firm2_test in iterprod(
        (dgl.RECConstants.INOUT, dgl.RECConstants.FIXED),
        [
            tuple(
                zip(
                    (
                        dgl.PCMConstants.UNI,
                        dgl.PCMConstants.BETA,
                        dgl.PCMConstants.EMPR,
                    ),
                    (
                        np.array((0, 1), dtype=np.float_),
                        np.array((10, 10), dtype=np.float_),
                        np.empty(2),
                    ),
                    strict=True,
                )
            )[_s]
            for _s in [0, 2]
        ],
        (dgl.FM2Constants.IID, dgl.FM2Constants.MNL),
    ):
        if recapture_spec_test == "proportional" and (
            pcm_dist_test_tup[0] != "Uniform" or pcm_dist_firm2_test == "MNL-dep"
        ):
            # When margins are specified as symmetric, then
            # recapture_spec must be proportional and
            # margins distributions must be iid;
            continue

        pcm_dist_type_test, pcm_dist_parms_test = pcm_dist_test_tup

        print()
        print(
            f"Simulated {_test_sel[0].capitalize()} rates by range of ∆HHI",
            f'recapture-rate calibrated, "{recapture_spec_test}"',
            f'Firm 2 margins, "{pcm_dist_firm2_test}"',
            f"and margins distributed {pcm_dist_type_test}{pcm_dist_parms_test}:",
            sep="; ",
        )

        ind_sample_spec = dgl.MarketSampleSpec(
            _sample_size,
            _clrenf_parm_vec[0],
            dgl.PRIConstants.SYM,
            share_spec=(
                dgl.SHRConstants.UNI,
                recapture_spec_test,
                dgl.DEFAULT_EMPTY_ARRAY,
            ),
            pcm_spec=(pcm_dist_type_test, pcm_dist_firm2_test, pcm_dist_parms_test),
        )

        clrenf_cnts_kwargs = {
            "sim_clrenf_sel": _test_sel,
            "save_data_to_file": _save_data_to_file,
        }

        start_time = datetime.now()
        (
            clrenf_rate_sim_byfirmcount_array,
            clrenf_rate_sim_bydelta_array,
            clrenf_rate_sim_byconczone_array,
        ) = gtl.sim_clrenf_cnts_ll(
            _clrenf_parm_vec, ind_sample_spec, clrenf_cnts_kwargs
        )
        run_duration = datetime.now() - start_time
        print(
            f"Simulation completed in {run_duration / timedelta(seconds=1):.6f} secs.",
            f"on {ind_sample_spec.sample_size:,d} draws",
            sep=", ",
        )

        stats_hdr_list, stats_dat_list = clstl.latex_tbl_clrenf_stats_1dim(
            clrenf_rate_sim_bydelta_array,
            return_type_sel=clstl.StatsReturnSelector.RPT,
            sort_order=clstl.SortSelector.REV,
        )
        stats_teststr_val = "".join([
            "{} & {} {}".format(
                stats_hdr_list[g],
                " & ".join(stats_dat_list[g][:-2]),  # [:-2]
                clstl.ltx_array_lineend,
            )
            for g in range(len(stats_hdr_list))
        ])
        print(stats_teststr_val)
        del stats_hdr_list, stats_dat_list, stats_teststr_val
        del pcm_dist_test_tup, pcm_dist_firm2_test, recapture_spec_test
        del pcm_dist_type_test, pcm_dist_parms_test

    if save_data_to_file_flag:
        _h5_datafile.close()


if __name__ == "__main__":
    analyze_clrenf_data(10**7, 2023, tests_of_interest[1], save_data_to_file_flag=False)
