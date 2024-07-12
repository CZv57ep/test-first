"""
Methods to generate data for analyzing merger enforcement policy.

"""

from __future__ import annotations

from attrs import define
from numpy.random import SeedSequence

from .. import VERSION  # noqa: TID252
from ..core.guidelines_boundaries import HMGThresholds  # noqa: TID252
from . import MarketSpec, UPPTestRegime
from .data_generation import gen_market_sample
from .upp_tests import SaveData, invres_cnts, save_data_to_hdf5, sim_invres_cnts_ll

__version__ = VERSION


@define(slots=False)
class MarketSample(MarketSpec):
    def generate_sample(
        self,
        /,
        *,
        sample_size: int = 10**6,
        seed_seq_list: list[SeedSequence] | None,
        nthreads: int,
        save_data_to_file: SaveData = False,
    ) -> None:
        self.data = gen_market_sample(
            self,
            sample_size=sample_size,
            seed_seq_list=seed_seq_list,
            nthreads=nthreads,
        )
        _invalid_array_names = (
            ("fcounts", "choice_prob_outgd", "nth_firm_share", "hhi_post")
            if self.share_spec.dist_type == "Uniform"
            else ()
        )
        if save_data_to_file:
            save_data_to_hdf5(
                self.data,
                excluded_attrs=_invalid_array_names,
                save_data_to_file=save_data_to_file,
            )

    def estimate_invres_counts(
        self,
        _invres_parm_vec: HMGThresholds,
        _upp_test_regime: UPPTestRegime,
        /,
        *,
        sample_size: int = 10**6,
        seed_seq_list: list[SeedSequence] | None,
        nthreads: int,
        save_data_to_file: SaveData = False,
    ) -> None:
        if getattr(self, "market_data_sample", None) is None:
            self.invres_counts = sim_invres_cnts_ll(
                self,
                _invres_parm_vec,
                _upp_test_regime,
                save_data_to_file=save_data_to_file,
                sample_size=sample_size,
                seed_seq_list=seed_seq_list,
                nthreads=nthreads,
            )
        else:
            self.invres_counts = invres_cnts(
                self.data, _invres_parm_vec, _upp_test_regime
            )
        if save_data_to_file:
            save_data_to_hdf5(self.invres_counts, save_data_to_file=save_data_to_file)
