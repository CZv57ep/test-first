"""
Methods to generate data for analyzing merger enforcement policy.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from attrs import Attribute, define, field, validators
from joblib import Parallel, cpu_count, delayed  # type: ignore
from numpy.random import SeedSequence

from .. import VERSION, RECTypes  # noqa: TID252  # noqa
from ..core import guidelines_boundaries as gbl  # noqa: TID252
from ..core.guidelines_boundaries import HMGThresholds  # noqa: TID252
from . import (
    FM2Constants,
    MarketDataSample,
    PCMDistributions,
    PCMSpec,
    PriceSpec,
    ShareSpec,
    SHRDistributions,
    SSZConstants,
    UPPTestRegime,
    UPPTestsCounts,
)
from .data_generation_functions import (
    gen_divr_array,
    gen_margin_price_data,
    gen_share_data,
    parse_seed_seq_list,
)
from .upp_tests import SaveData, enf_cnts, save_data_to_hdf5

__version__ = VERSION


class INVRESCntsArgs(TypedDict, total=False):
    "Keyword arguments of function, :func:`MarketSample.sim_enf_cnts`"

    sample_size: int
    seed_seq_list: list[SeedSequence] | None
    nthreads: int
    save_data_to_file: SaveData
    saved_array_name_suffix: str


@define
class MarketSample:
    """Parameter specification for market data generation."""

    share_spec: ShareSpec = field(
        kw_only=True,
        default=ShareSpec(SHRDistributions.UNI, None, None, RECTypes.INOUT, 0.8),
        validator=validators.instance_of(ShareSpec),
    )
    """Market-share specification, see :class:`ShareSpec`"""

    pcm_spec: PCMSpec = field(
        kw_only=True, default=PCMSpec(PCMDistributions.UNI, None, FM2Constants.IID)
    )
    """Margin specification, see :class:`PCMSpec`"""

    @pcm_spec.validator
    def _check_pcm(self, _a: Attribute[PCMSpec], _v: PCMSpec, /) -> None:
        if (
            self.share_spec.recapture_form == RECTypes.FIXED
            and _v.firm2_pcm_constraint == FM2Constants.MNL
        ):
            raise ValueError(
                f'Specification of "recapture_form", "{self.share_spec.recapture_form}" '
                "requires Firm 2 margin must have property, "
                f'"{FM2Constants.IID}" or "{FM2Constants.SYM}".'
            )

    price_spec: PriceSpec = field(
        kw_only=True, default=PriceSpec.SYM, validator=validators.instance_of(PriceSpec)
    )
    """Price specification, see :class:`PriceSpec`"""

    hsr_filing_test_type: SSZConstants = field(
        kw_only=True,
        default=SSZConstants.ONE,
        validator=validators.instance_of(SSZConstants),
    )
    """Method for modeling HSR filing threholds, see :class:`SSZConstants`"""

    data: MarketDataSample = field(default=None)

    enf_counts: UPPTestsCounts = field(default=None)

    def gen_market_sample(
        self,
        /,
        *,
        sample_size: int = 10**6,
        seed_seq_list: Sequence[SeedSequence] | None = None,
        nthreads: int = 16,
    ) -> MarketDataSample:
        """
        Generate share, diversion ratio, price, and margin data for MarketSpec.


        Parameters
        ----------
        _mkt_sample_spec
            class specifying parameters for data generation, see :class:`mergeron.gen.MarketSpec`
        sample_size
            number of draws to generate
        seed_seq_list
            tuple of SeedSequences to ensure replicable data generation with
            appropriately independent random streams
        nthreads
            optionally specify the number of CPU threads for the PRNG

        Returns
        -------
            Merging firms' shares, margins, etc. for each hypothetical  merger
            in the sample

        """

        _recapture_form = self.share_spec.recapture_form
        _recapture_rate = self.share_spec.recapture_rate
        _dist_type_mktshr = self.share_spec.dist_type
        _dist_firm2_pcm = self.pcm_spec.firm2_pcm_constraint
        _hsr_filing_test_type = self.hsr_filing_test_type

        (
            _mktshr_rng_seed_seq,
            _pcm_rng_seed_seq,
            _fcount_rng_seed_seq,
            _pr_rng_seed_seq,
        ) = parse_seed_seq_list(seed_seq_list, _dist_type_mktshr, self.price_spec)

        _shr_sample_size = 1.0 * sample_size
        # Scale up sample size to offset discards based on specified criteria
        _shr_sample_size *= _hsr_filing_test_type
        if _dist_firm2_pcm == FM2Constants.MNL:
            _shr_sample_size *= SSZConstants.MNL_DEP
        _shr_sample_size = int(_shr_sample_size)

        # Generate share data
        _mktshr_data = gen_share_data(
            _shr_sample_size,
            self.share_spec,
            _fcount_rng_seed_seq,
            _mktshr_rng_seed_seq,
            nthreads,
        )

        _mktshr_array, _fcounts, _aggregate_purchase_prob, _nth_firm_share = (
            getattr(_mktshr_data, _f)
            for _f in (
                "mktshr_array",
                "fcounts",
                "aggregate_purchase_prob",
                "nth_firm_share",
            )
        )

        # Generate merging-firm price and PCM data
        _margin_data, _price_data = gen_margin_price_data(
            _mktshr_array[:, :2],
            _nth_firm_share,
            _aggregate_purchase_prob,
            self.pcm_spec,
            self.price_spec,
            self.hsr_filing_test_type,
            _pcm_rng_seed_seq,
            _pr_rng_seed_seq,
            nthreads,
        )

        _price_array, _hsr_filing_test = (
            getattr(_price_data, _f) for _f in ("price_array", "hsr_filing_test")
        )

        _pcm_array, _mnl_test_rows = (
            getattr(_margin_data, _f) for _f in ("pcm_array", "mnl_test_array")
        )

        _mnl_test_rows = _mnl_test_rows * _hsr_filing_test
        _s_size = sample_size  # originally-specified sample size
        if _dist_firm2_pcm == FM2Constants.MNL:
            _mktshr_array = _mktshr_array[_mnl_test_rows][:_s_size]
            _pcm_array = _pcm_array[_mnl_test_rows][:_s_size]
            _price_array = _price_array[_mnl_test_rows][:_s_size]
            _fcounts = _fcounts[_mnl_test_rows][:_s_size]
            _aggregate_purchase_prob = _aggregate_purchase_prob[_mnl_test_rows][
                :_s_size
            ]
            _nth_firm_share = _nth_firm_share[_mnl_test_rows][:_s_size]

        # Calculate diversion ratios
        _divr_array = gen_divr_array(
            _recapture_form,
            _recapture_rate,
            _mktshr_array[:, :2],
            _aggregate_purchase_prob,
        )

        del _mnl_test_rows, _s_size

        _frmshr_array = _mktshr_array[:, :2]
        _hhi_delta = np.einsum("ij,ij->i", _frmshr_array, _frmshr_array[:, ::-1])[
            :, None
        ]

        _hhi_post = (
            _hhi_delta + np.einsum("ij,ij->i", _mktshr_array, _mktshr_array)[:, None]
        )

        return MarketDataSample(
            _frmshr_array,
            _pcm_array,
            _price_array,
            _fcounts,
            _aggregate_purchase_prob,
            _nth_firm_share,
            _divr_array,
            _hhi_post,
            _hhi_delta,
        )

    def generate_sample(
        self,
        /,
        *,
        sample_size: int = 10**6,
        seed_seq_list: list[SeedSequence] | None,
        nthreads: int,
        save_data_to_file: SaveData = False,
        saved_array_name_suffix: str = "",
    ) -> None:
        """Generate market data"""

        self.data = self.gen_market_sample(
            sample_size=sample_size, seed_seq_list=seed_seq_list, nthreads=nthreads
        )

        _invalid_array_names = (
            ("fcounts", "choice_prob_outgd", "nth_firm_share", "hhi_post")
            if self.share_spec.dist_type == "Uniform"
            else ()
        )

        save_data_to_hdf5(
            self.data,
            saved_array_name_suffix=saved_array_name_suffix,
            excluded_attrs=_invalid_array_names,
            save_data_to_file=save_data_to_file,
        )

    def sim_enf_cnts(
        self,
        _upp_test_parms: gbl.HMGThresholds,
        _sim_test_regime: UPPTestRegime,
        /,
        *,
        sample_size: int = 10**6,
        seed_seq_list: list[SeedSequence] | None = None,
        nthreads: int = 16,
        save_data_to_file: SaveData = False,
        saved_array_name_suffix: str = "",
    ) -> UPPTestsCounts:
        """Generate market data and etstimate UPP test counts on same.

        Parameters
        ----------

        _mkt_sample_spec
            Configuration to use for generating sample data to test

        _upp_test_parms
            Guidelines thresholds for testing UPP and related statistics

        _sim_test_regime
            Configuration to use for testing; UPPTestsRegime object
            specifying whether investigation results in enforcement, clearance,
            or both; and aggregation methods used for GUPPI and diversion ratio
            measures

        sample_size
            Number of draws to generate

        seed_seq_list
            List of seed sequences, to assure independent samples in each thread

        nthreads
            Number of parallel processes to use

        save_data_to_file
            Whether to save data to an HDF5 file, and where to save it

        saved_array_name_suffix
            Suffix to add to the array names in the HDF5 file

        Returns
        -------
            UPPTestCounts ojbect with  of test counts by firm count, ΔHHI and concentration zone

        """

        _market_data_sample = self.gen_market_sample(
            sample_size=sample_size, seed_seq_list=seed_seq_list, nthreads=nthreads
        )

        _invalid_array_names = (
            ("fcounts", "choice_prob_outgd", "nth_firm_share", "hhi_post")
            if self.share_spec.dist_type == "Uniform"
            else ()
        )

        save_data_to_hdf5(
            _market_data_sample,
            saved_array_name_suffix=saved_array_name_suffix,
            excluded_attrs=_invalid_array_names,
            save_data_to_file=save_data_to_file,
        )

        _upp_test_arrays = enf_cnts(
            _market_data_sample, _upp_test_parms, _sim_test_regime
        )

        save_data_to_hdf5(
            _upp_test_arrays,
            saved_array_name_suffix=saved_array_name_suffix,
            save_data_to_file=save_data_to_file,
        )

        return _upp_test_arrays

    def sim_enf_cnts_ll(
        self,
        _enf_parm_vec: gbl.HMGThresholds,
        _sim_test_regime: UPPTestRegime,
        /,
        *,
        sample_size: int = 10**6,
        seed_seq_list: list[SeedSequence] | None = None,
        nthreads: int = 16,
        save_data_to_file: SaveData = False,
        saved_array_name_suffix: str = "",
    ) -> UPPTestsCounts:
        """A function to parallelize data-generation and testing

        The parameters `_sim_enf_cnts_kwargs` are passed unaltered to
        the parent function, `sim_enf_cnts()`, except that, if provided,
        `seed_seq_list` is used to spawn a seed sequence for each thread,
        to assure independent samples in each thread, and `nthreads` defines
        the number of parallel processes used. The number of draws in
        each thread may be tuned, by trial and error, to the amount of
        memory (RAM) available.

        Parameters
        ----------

        _enf_parm_vec
            Guidelines thresholds to test against

        _mkt_sample_spec
            Configuration to use for generating sample data to test

        _sim_test_regime
            Configuration to use for testing

        saved_array_name_suffix
            Suffix to add to the array names in the HDF5 file

        save_data_to_file
            Whether to save data to an HDF5 file, and where to save it

        sample_size
            Number of draws to simulate

        seed_seq_list
            List of seed sequences, to assure independent samples in each thread

        nthreads
            Number of parallel processes to use

        Returns
        -------
            Arrays of UPPTestCounts

        """
        _sample_sz = sample_size
        _subsample_sz = 10**6
        _iter_count = (
            int(_sample_sz / _subsample_sz) if _subsample_sz < _sample_sz else 1
        )
        _thread_count = cpu_count()

        if (
            self.share_spec.recapture_form != RECTypes.OUTIN
            and self.share_spec.recapture_rate != _enf_parm_vec.rec
        ):
            raise ValueError(
                "{} {} {}".format(
                    f"Recapture rate from market sample spec, {self.share_spec.recapture_rate}",
                    f"must match the value, {_enf_parm_vec.rec}",
                    "the guidelines thresholds vector.",
                )
            )

        _rng_seed_seq_list = [None] * _iter_count
        if seed_seq_list:
            _rng_seed_seq_list = list(
                zip(*[g.spawn(_iter_count) for g in seed_seq_list], strict=True)  # type: ignore
            )

        _sim_enf_cnts_kwargs: INVRESCntsArgs = INVRESCntsArgs({
            "sample_size": _subsample_sz,
            "save_data_to_file": save_data_to_file,
            "nthreads": nthreads,
        })

        _res_list = Parallel(n_jobs=_thread_count, prefer="threads")(
            delayed(self.sim_enf_cnts)(
                _enf_parm_vec,
                _sim_test_regime,
                **_sim_enf_cnts_kwargs,
                saved_array_name_suffix=f"{saved_array_name_suffix}_{_iter_id:0{2 + int(np.ceil(np.log10(_iter_count)))}d}",
                seed_seq_list=_rng_seed_seq_list_ch,
            )
            for _iter_id, _rng_seed_seq_list_ch in enumerate(_rng_seed_seq_list)
        )

        _res_list_stacks = UPPTestsCounts(*[
            np.stack([getattr(_j, _k) for _j in _res_list])
            for _k in ("by_firm_count", "by_delta", "by_conczone")
        ])
        upp_test_results = UPPTestsCounts(*[
            np.column_stack((
                (_gv := getattr(_res_list_stacks, _g))[0, :, :_h],
                np.einsum("ijk->jk", np.int64(1) * _gv[:, :, _h:]),
            ))
            for _g, _h in zip(
                _res_list_stacks.__dataclass_fields__.keys(), [1, 1, 3], strict=True
            )
        ])
        del _res_list, _res_list_stacks

        return upp_test_results

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
        """Estimate enforcement counts

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

        if self.data is None:
            self.enf_counts = self.sim_enf_cnts_ll(
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
