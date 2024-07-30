"""
Methods to generate market data, including shares price, marginsm, and diversion ratios
for analyzing merger enforcement policy.

"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.random import SeedSequence
from numpy.typing import NDArray

from .. import VERSION, RECConstants  # noqa: TID252
from . import (
    EMPTY_ARRAY_DEFAULT,
    FM2Constants,
    MarketDataSample,
    MarketSpec,
    PriceConstants,
    SHRConstants,
    SSZConstants,
)
from ._data_generation_functions import _gen_margin_price_data, _gen_share_data

__version__ = VERSION


class SeedSequenceData(NamedTuple):
    mktshr_rng_seed_seq: SeedSequence
    pcm_rng_seed_seq: SeedSequence
    fcount_rng_seed_seq: SeedSequence | None
    pr_rng_seed_seq: SeedSequence | None


def gen_market_sample(
    _mkt_sample_spec: MarketSpec,
    /,
    *,
    sample_size: int = 10**6,
    seed_seq_list: list[SeedSequence] | None = None,
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
    _mktshr_data = _gen_share_data(
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
    _margin_data, _price_data = _gen_margin_price_data(
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
    _sseq_list: list[SeedSequence] | None,
    _mktshr_dist_type: SHRConstants,
    _price_spec: PriceConstants,
    /,
) -> SeedSequenceData:
    """Initialize RNG seed sequences to ensure independence of distinct random streams.

    The tuple of SeedSequences, is parsed in the following order
    for generating the relevant random variates:
    1.) quantity shares
    2.) price-cost margins
    3.) firm-counts, if :code:`MarketSpec.share_spec.dist_type` is a Dirichlet distribution
    4.) prices, if :code:`MarketSpec.price_spec ==`:attr:`mergeron.gen.PriceConstants.ZERO`.



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

    if _price_spec == PriceConstants.ZERO:
        _pr_rng_seed_seq = _sseq_list.pop() if _sseq_list else SeedSequence(pool_size=8)

    if _mktshr_dist_type == SHRConstants.UNI:
        _fcount_rng_seed_seq = None
        _seed_count = 2
        _mktshr_rng_seed_seq, _pcm_rng_seed_seq = (
            _sseq_list[:_seed_count]
            if _sseq_list
            else (SeedSequence(pool_size=8) for _ in range(_seed_count))
        )
    else:
        _seed_count = 3
        (_mktshr_rng_seed_seq, _pcm_rng_seed_seq, _fcount_rng_seed_seq) = (
            _sseq_list[:_seed_count]
            if _sseq_list
            else (SeedSequence(pool_size=8) for _ in range(_seed_count))
        )

    return SeedSequenceData(
        _mktshr_rng_seed_seq, _pcm_rng_seed_seq, _fcount_rng_seed_seq, _pr_rng_seed_seq
    )


def gen_divr_array(
    _recapture_form: RECConstants,
    _recapture_rate: float | None,
    _frmshr_array: NDArray[np.float64],
    _aggregate_purchase_prob: NDArray[np.float64] = EMPTY_ARRAY_DEFAULT,
    /,
) -> NDArray[np.float64]:
    """
    Given merging-firm shares and related parameters, return diverion ratios.

    If recapture is specified as :attr:`mergeron.RECConstants.OUTIN`, then the
    choice-probability for the outside good must be supplied.

    Parameters
    ----------
    _recapture_form
        Enum specifying Fixed (proportional), Inside-out, or Outside-in

    _recapture_rate
        If recapture is proportional or inside-out, the recapture rate
        for the firm with the smaller share.

    _frmshr_array
        Merging-firm shares.

    _aggregate_purchase_prob
        1 minus probability that the outside good is chosen; converts
        market shares to choice probabilities by multiplication.

    Returns
    -------
        Merging-firm diversion ratios for mergers in the sample.

    """

    _divr_array: NDArray[np.float64]
    if _recapture_form == RECConstants.FIXED:
        _divr_array = _recapture_rate * _frmshr_array[:, ::-1] / (1 - _frmshr_array)  # type: ignore

    else:
        _purchprob_array = _aggregate_purchase_prob * _frmshr_array
        _divr_array = _purchprob_array[:, ::-1] / (1 - _purchprob_array)

    _divr_assert_test = (
        (np.round(np.einsum("ij->i", _frmshr_array), 15) == 1)
        | (np.argmin(_frmshr_array, axis=1) == np.argmax(_divr_array, axis=1))
    )[:, None]
    if not all(_divr_assert_test):
        raise ValueError(
            "{} {} {} {}".format(
                "Data construction fails tests:",
                "the index of min(s_1, s_2) must equal",
                "the index of max(d_12, d_21), for all draws.",
                "unless frmshr_array sums to 1.00.",
            )
        )

    return _divr_array
