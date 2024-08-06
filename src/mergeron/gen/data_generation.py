"""
Methods to generate market data, including shares price, marginsm, and diversion ratios
for analyzing merger enforcement policy.

"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.random import SeedSequence

from .. import VERSION  # noqa: TID252
from . import (
    FM2Constants,
    MarketDataSample,
    PriceSpec,
    SeedSequenceData,
    SHRDistributions,
    SSZConstants,
)
from .data_generation_functions import (
    gen_divr_array,
    gen_margin_price_data,
    gen_share_data,
)
from .market_sample import MarketSample

__version__ = VERSION


def gen_market_sample(
    _mkt_sample_spec: MarketSample,
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

    _recapture_form = _mkt_sample_spec.share_spec.recapture_form
    _recapture_rate = _mkt_sample_spec.share_spec.recapture_rate
    _dist_type_mktshr = _mkt_sample_spec.share_spec.dist_type
    _dist_firm2_pcm = _mkt_sample_spec.pcm_spec.firm2_pcm_constraint
    _hsr_filing_test_type = _mkt_sample_spec.hsr_filing_test_type

    (
        _mktshr_rng_seed_seq,
        _pcm_rng_seed_seq,
        _fcount_rng_seed_seq,
        _pr_rng_seed_seq,
    ) = parse_seed_seq_list(
        seed_seq_list, _dist_type_mktshr, _mkt_sample_spec.price_spec
    )

    _shr_sample_size = 1.0 * sample_size
    # Scale up sample size to offset discards based on specified criteria
    _shr_sample_size *= _hsr_filing_test_type
    if _dist_firm2_pcm == FM2Constants.MNL:
        _shr_sample_size *= SSZConstants.MNL_DEP
    _shr_sample_size = int(_shr_sample_size)

    # Generate share data
    _mktshr_data = gen_share_data(
        _shr_sample_size,
        _mkt_sample_spec.share_spec,
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
        _mkt_sample_spec.pcm_spec,
        _mkt_sample_spec.price_spec,
        _mkt_sample_spec.hsr_filing_test_type,
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
        _aggregate_purchase_prob = _aggregate_purchase_prob[_mnl_test_rows][:_s_size]
        _nth_firm_share = _nth_firm_share[_mnl_test_rows][:_s_size]

    # Calculate diversion ratios
    _divr_array = gen_divr_array(
        _recapture_form, _recapture_rate, _mktshr_array[:, :2], _aggregate_purchase_prob
    )

    del _mnl_test_rows, _s_size

    _frmshr_array = _mktshr_array[:, :2]
    _hhi_delta = np.einsum("ij,ij->i", _frmshr_array, _frmshr_array[:, ::-1])[:, None]

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


def parse_seed_seq_list(
    _sseq_list: Sequence[SeedSequence] | None,
    _mktshr_dist_type: SHRDistributions,
    _price_spec: PriceSpec,
    /,
) -> SeedSequenceData:
    """Initialize RNG seed sequences to ensure independence of distinct random streams.

    The tuple of SeedSequences, is parsed in the following order
    for generating the relevant random variates:
    1.) quantity shares
    2.) price-cost margins
    3.) firm-counts, if :code:`MarketSpec.share_spec.dist_type` is a Dirichlet distribution
    4.) prices, if :code:`MarketSpec.price_spec ==`:attr:`mergeron.gen.PriceSpec.ZERO`.



    Parameters
    ----------
    _sseq_list
        List of RNG seed sequences

    _mktshr_dist_type
        Market share distribution type

    _price_spec
        Price specification

    Returns
    -------
        Seed sequence data

    """
    _fcount_rng_seed_seq: SeedSequence | None = None
    _pr_rng_seed_seq: SeedSequence | None = None

    if _price_spec == PriceSpec.ZERO:
        _sseq_list, _pr_rng_seed_seq = (
            (_sseq_list[:-1], _sseq_list[-1])
            if _sseq_list
            else (None, SeedSequence(pool_size=8))
        )

    _seed_count = 2 if _mktshr_dist_type == SHRDistributions.UNI else 3

    if _sseq_list:
        if len(_sseq_list) < _seed_count:
            raise ValueError(
                f"seed sequence list must contain {_seed_count} seed sequences"
            )
    else:
        _sseq_list = tuple(SeedSequence(pool_size=8) for _ in range(_seed_count))

    (_mktshr_rng_seed_seq, _pcm_rng_seed_seq, _fcount_rng_seed_seq) = (
        _sseq_list if _seed_count == 3 else (*_sseq_list, None)  # type: ignore
    )

    return SeedSequenceData(
        _mktshr_rng_seed_seq, _pcm_rng_seed_seq, _fcount_rng_seed_seq, _pr_rng_seed_seq
    )
