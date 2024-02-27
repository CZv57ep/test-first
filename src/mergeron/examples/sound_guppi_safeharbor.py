from datetime import datetime, timedelta
from itertools import product as iterprod
from pathlib import Path
from typing import Literal

import numpy as np
import tables as ptb  # type: ignore
from attrs import fields as attrs_fields

import mergeron.core.guidelines_boundaries as gbl
import mergeron.gen.data_generation as dgl
import mergeron.gen.investigations_stats as isl
import mergeron.gen.upp_tests as utl
from mergeron import DATA_DIR
from mergeron.core.pseudorandom_numbers import DIST_PARMS_DEFAULT
from mergeron.gen import (
    INVResolution,
    MarketSampleSpec,
    PCMConstants,
    PCMSpec,
    ShareSpec,
    SHRConstants,
    UPPAggrSelector,
    UPPTestRegime,
)

PROG_PATH = Path(__file__)

tests_of_interest: tuple[UPPTestRegime, ...] = (
    UPPTestRegime(INVResolution.CLRN, UPPAggrSelector.MAX, UPPAggrSelector.MAX),
    UPPTestRegime(INVResolution.ENFT, UPPAggrSelector.MIN, UPPAggrSelector.MIN),
)


def analyze_invres_data(
    _sample_size: int = 10**6,
    _hmg_std_pub_year: Literal[1992, 2010, 2023] = 1992,
    _test_sel: UPPTestRegime = tests_of_interest[1],
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

    _hmg_pub_year
        Guidelines version for ∆HHI standard

    _test_regime
        Specifies analysis of enforcement rates or, alternatively, clearance rates

    save_data_to_file_flag
        If True, simulated data are save to file (hdf5 format)

    """
    _invres_parm_vec = gbl.GuidelinesThresholds(_hmg_pub_year).presumption

    _save_data_to_file: utl.SaveData = False
    if save_data_to_file_flag:
        _h5_path = DATA_DIR / PROG_PATH.with_suffix(".h5").name
        _, _h5_file, _h5_group = utl.initialize_hd5(
            h5_path, _hmg_pub_year, _test_regime
        )  # type: ignore

        _h5_group_name = "invres_yr{}_res{}_priagg{}_secagg{}".format(
            _hmg_pub_year,
            *(
                getattr(_test_regime, _f.name).name
                for _f in attrs_fields(type(_test_regime))
            ),
        )
        _h5_group = _h5_file.create_group(
            _h5_group, f"invres_{_hmg_pub_year}", f"{_invres_parm_vec}"
        )
        _save_data_to_file = (True, _h5_file, _h5_group)

    # ##
    #   Print summaries of intrinsic clearance/enforcement rates by ∆HHI,
    #   with asymmetric margins
    #  ##
    for _recapture_spec_test, _pcm_dist_test, _pcm_dist_firm2_test in iterprod(
        (dgl.RECConstants.INOUT, dgl.RECConstants.FIXED),
        [
            tuple(
                zip(
                    (PCMConstants.UNI, PCMConstants.BETA, PCMConstants.EMPR),
                    (
                        np.array((0, 1), dtype=np.float64),
                        np.array((10, 10), dtype=np.float64),
                        None,
                    ),
                    strict=True,
                )
            )[_s]
            for _s in [0, 2]
        ],
        (dgl.FM2Constants.IID, dgl.FM2Constants.MNL),
    ):
        if _recapture_spec_test == "proportional" and (
            _pcm_dist_test[0] != "Uniform" or _pcm_dist_firm2_test == "MNL-dep"
        ):
            continue
            # When margins are specified as symmetric, then
            # recapture_spec must be proportional and
            # margins distributions must be iid;

        _pcm_dist_type_test, _pcm_dist_parms_test = _pcm_dist_test

        print()
        print(
            f"Simulated {_test_regime.resolution.capitalize()} rates by range of ∆HHI",
            f'recapture-rate calibrated, "{_recapture_spec_test}"',
            f'Firm 2 margins, "{_pcm_dist_firm2_test}"',
            f"and margins distributed as, {_pcm_dist_type_test}{
                _pcm_dist_parms_test if _pcm_dist_type_test.name == "BETA" else ""
            }:",
            sep="; ",
        )

        _mkt_sample_spec = MarketSampleSpec(
            _sample_size,
            _invres_parm_vec.rec,
            share_spec=ShareSpec(
                _recapture_spec_test, SHRConstants.UNI, DIST_PARMS_DEFAULT, None
            ),
            pcm_spec=PCMSpec(
                _pcm_dist_type_test, _pcm_dist_firm2_test, _pcm_dist_parms_test
            ),
        )

        if _save_data_to_file:
            _h5_file.flush()

            _h5_subgrp_name = "invres_rec{}_pcm{}_fm2res{}".format(
                _recapture_spec_test.name,
                _pcm_dist_type_test.name,
                _pcm_dist_firm2_test.name,
            )

            _h5_subgroup = _h5_file.create_group(
                _h5_group, _h5_subgrp_name, title=f"{_mkt_sample_spec}"
            )
            _save_data_to_file = (True, _h5_file, _h5_subgroup)

        _invres_cnts_kwargs = utl.IVNRESCntsArgs(
            sim_test_regime=_test_regime, save_data_to_file=_save_data_to_file
        )

        _start_time = datetime.now()

        upp_test_counts = utl.sim_invres_cnts_ll(
            _invres_parm_vec, _mkt_sample_spec, _invres_cnts_kwargs
        )
        _run_duration = datetime.now() - _start_time
        print(
            f"Simulation completed in {_run_duration / timedelta(seconds=1):.6f} secs.",
            f"on {_mkt_sample_spec.sample_size:,d} draws",
            sep=", ",
        )

        _stats_hdr_list, _stats_dat_list = isl.latex_tbl_invres_stats_1dim(
            upp_test_counts.by_delta,
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
        del _pcm_dist_test, _pcm_dist_firm2_test, _recapture_spec_test
        del _pcm_dist_type_test, _pcm_dist_parms_test

    if _save_data_to_file:
        _h5_file.flush()
        _save_data_to_file[1].close()  # type: ignore


if __name__ == "__main__":
    analyze_invres_data(10**7, 2023, tests_of_interest[1], save_data_to_file_flag=True)
