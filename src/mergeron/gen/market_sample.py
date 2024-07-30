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
from .upp_tests import SaveData, enf_cnts, save_data_to_hdf5, sim_enf_cnts_ll

__version__ = VERSION


@define(slots=False)
class MarketSample(MarketSpec):
    def generate_sample(
        self,
        /,
        *,
        sample_size: int,
        seed_seq_list: list[SeedSequence] | None,
        nthreads: int,
        save_data_to_file: SaveData,
    ) -> None:
        """Generate market data

        Parameters
        ----------
        sample_size
            Size of the market sample drawn

        seed_seq_list
            List of :code:`numpy.random.SeedSequence` objects

        nthreads
            Number of threads to use

        save_data_to_file
            Save data to given HDF5 file, at specified group node

        Returns
        -------
        None

        Notes
        -----
        See documentation for :class:`mergeron.gen.data_generation.gen_market_sample`
        for more information, and :func:`mergeron.gen.data_generation.parse_seed_seq_list`
        on the specification of :code:`seed_seq_list`.

        """
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

    def estimate_enf_counts(
        self,
        _enf_parm_vec: HMGThresholds,
        _upp_test_regime: UPPTestRegime,
        /,
        *,
        sample_size: int = 10**6,
        seed_seq_list: list[SeedSequence] | None,
        nthreads: int,
        save_data_to_file: SaveData = False,
    ) -> None:
        """Generate market data

        Parameters
        ----------
        _enf_parm_vec
            Threshold values for various Guidelines criteria

        _upp_test_regime
            Specifies whether to analyze enforcement, clearance, or both
            and the GUPPI and diversion ratio aggregators employed, with
            default being to analyze enforcement based on the maximum
            merging-firm GUPPI and maximum diversion ratio between the
            merging firms

        sample_size
            Size of the market sample drawn

        seed_seq_list
            List of :code:`numpy.random.SeedSequence` objects

        nthreads
            Number of threads to use

        save_data_to_file
            Save data to given HDF5 file, at specified group node

        Returns
        -------
        None

        Notes
        -----
        See documentation for :class:`mergeron.gen.MarketSpec` for details on specifying
        how shares, margins, prices, and diversion ratios are generated, and whether to restrict
        the sample to draws representing mergers that meet the HSR filing requirements. See
        :class:`mergeron.gen.MarketDataSample` on the sample data generated; see,
        :func:`mergeron.gen.data_generation.parse_seed_seq_list` on
        the specification of :code:`seed_seq_list`.

        """

        if getattr(self, "market_data_sample", None) is None:
            self.enf_counts = sim_enf_cnts_ll(
                self,
                _enf_parm_vec,
                _upp_test_regime,
                sample_size=sample_size,
                seed_seq_list=seed_seq_list,
                nthreads=nthreads,
                save_data_to_file=save_data_to_file,
            )
        else:
            self.enf_counts = enf_cnts(self.data, _enf_parm_vec, _upp_test_regime)
            if save_data_to_file:
                save_data_to_hdf5(self.enf_counts, save_data_to_file=save_data_to_file)
