import re
from datetime import datetime, timedelta
from itertools import product as iterprod
from pathlib import Path
from typing import Literal

import numpy as np
import tables as ptb  # type: ignore

import mergeron.core.guidelines_standards as gsf
import mergeron.gen.data_generation as dgl
import mergeron.gen.guidelines_tests as gtl
import mergeron.gen.investigations_stats as isl
from mergeron import DATA_DIR

tests_of_interest: tuple[gtl.UPPTestRegime, ...] = (
    (isl.PolicySelector.CLRN, gtl.GUPPIWghtngSelector.MAX, gtl.GUPPIWghtngSelector.MAX),
    (isl.PolicySelector.ENFT, gtl.GUPPIWghtngSelector.MIN, gtl.GUPPIWghtngSelector.MIN),
)

PROG_PATH = Path(__file__)


def analyze_invres_data(
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
    _invres_parm_vec = gsf.GuidelinesStandards(_hmg_std_pub_year).presumption

    _save_data_to_file: gtl.SaveData = False
    if save_data_to_file_flag:
        _h5_hier_pat = re.compile(r"\W")
        _blosc_filters = ptb.Filters(
            complevel=3, complib="blosc:lz4hc", fletcher32=True
        )
        _h5_datafile = ptb.open_file(
            DATA_DIR / PROG_PATH.with_suffix(".h5").name,
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
    for _recapture_spec_test, _pcm_dist_test_tup, _pcm_dist_firm2_test in iterprod(
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
                        np.array((0, 1), dtype=np.float64),
                        np.array((10, 10), dtype=np.float64),
                        np.empty(2),
                    ),
                    strict=True,
                )
            )[_s]
            for _s in [0, 2]
        ],
        (dgl.FM2Constants.IID, dgl.FM2Constants.MNL),
    ):
        if _recapture_spec_test == "proportional" and (
            _pcm_dist_test_tup[0] != "Uniform" or _pcm_dist_firm2_test == "MNL-dep"
        ):
            # When margins are specified as symmetric, then
            # recapture_spec must be proportional and
            # margins distributions must be iid;
            continue

        _pcm_dist_type_test, _pcm_dist_parms_test = _pcm_dist_test_tup

        print()
        print(
            f"Simulated {_test_sel[0].capitalize()} rates by range of ∆HHI",
            f'recapture-rate calibrated, "{_recapture_spec_test}"',
            f'Firm 2 margins, "{_pcm_dist_firm2_test}"',
            f"and margins distributed {_pcm_dist_type_test}{_pcm_dist_parms_test}:",
            sep="; ",
        )

        _ind_sample_spec = dgl.MarketSampleSpec(
            _sample_size,
            _invres_parm_vec.rec,
            dgl.PRIConstants.SYM,
            share_spec=dgl.ShareSpec(
                _recapture_spec_test,
                dgl.SHRConstants.UNI,
                dgl.EMPTY_ARRAY_DEFAULT,
                dgl.FCOUNT_WTS_DEFAULT,
            ),
            pcm_spec=dgl.PCMSpec(
                _pcm_dist_type_test, _pcm_dist_firm2_test, _pcm_dist_parms_test
            ),
        )

        _invres_cnts_kwargs = {
            "sim_test_regime": _test_sel,
            "save_data_to_file": _save_data_to_file,
        }

        _start_time = datetime.now()
        (
            _invres_rate_sim_byfirmcount_array,
            _invres_rate_sim_bydelta_array,
            _invres_rate_sim_byconczone_array,
        ) = gtl.sim_invres_cnts_ll(
            _invres_parm_vec, _ind_sample_spec, _invres_cnts_kwargs
        )
        _run_duration = datetime.now() - _start_time
        print(
            f"Simulation completed in {_run_duration / timedelta(seconds=1):.6f} secs.",
            f"on {_ind_sample_spec.sample_size:,d} draws",
            sep=", ",
        )

        _stats_hdr_list, _stats_dat_list = isl.latex_tbl_invres_stats_1dim(
            _invres_rate_sim_bydelta_array,
            return_type_sel=isl.StatsReturnSelector.RPT,
            sort_order=isl.SortSelector.REV,
        )
        _stats_teststr_val = "".join([
            "{} & {} {}".format(
                _stats_hdr_list[g],
                " & ".join(_stats_dat_list[g][:-2]),  # [:-2]
                isl.LTX_ARRAY_LINEEND,
            )
            for g in range(len(_stats_hdr_list))
        ])
        print(_stats_teststr_val)
        del _stats_hdr_list, _stats_dat_list, _stats_teststr_val
        del _pcm_dist_test_tup, _pcm_dist_firm2_test, _recapture_spec_test
        del _pcm_dist_type_test, _pcm_dist_parms_test

    if save_data_to_file_flag:
        _save_data_to_file[1].close()  # type: ignore


if __name__ == "__main__":
    analyze_invres_data(10**7, 2023, tests_of_interest[1], save_data_to_file_flag=False)
