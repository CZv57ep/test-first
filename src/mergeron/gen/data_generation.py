"""
Routines to generate data for analyzing merger enforcement policy.

"""

from __future__ import annotations

from importlib.metadata import version

import attrs
import numpy as np
from numpy.random import SeedSequence
from numpy.typing import NDArray

from .. import _PKG_NAME, RECConstants  # noqa: TID252
from . import (
    EMPTY_ARRAY_DEFAULT,
    TF,
    FM2Constants,
    MarketDataSample,
    MarketSampleSpec,
    PRIConstants,
    SHRConstants,
    SSZConstants,
)
from ._data_generation_functions_nonpublic import (
    _gen_market_shares_dirichlet,  # noqa: F401 easter-egg for external modules
    _gen_market_shares_uniform,  # noqa: F401 easter-egg for external modules
    _gen_pcm_data,
    _gen_pr_data,
    _gen_share_data,
)

__version__ = version(_PKG_NAME)


def gen_market_sample(
    _mkt_sample_spec: MarketSampleSpec,
    /,
    *,
    seed_seq_list: list[SeedSequence] | None = None,
    nthreads: int = 16,
) -> MarketDataSample:
    """
    Generate share, diversion ratio, price, and margin data based on supplied parameters

    Diversion ratios generated assuming share-proportionality, unless
    `recapture_spec` = "proportional", in which case both firms' recapture rate
    is set to `r_bar`.

    The tuple of SeedSequences, if specified, is parsed in the following order
    for generating the relevant random variates:
    1.) quantity shares
    2.) price-cost margins
    3.) firm-counts, from :code:`[2, 2 + len(firm_counts_weights)]`,
    weighted by :code:`firm_counts_weights`, where relevant
    4.) prices, if :code:`pr_sym_spec == PRIConstants.ZERO`.

    Parameters
    ----------
    _mkt_sample_spec
        class specifying parameters for data generation
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

    _mkt_sample_spec = _mkt_sample_spec or MarketSampleSpec()

    _recapture_spec = _mkt_sample_spec.share_spec.recapture_spec
    _dist_type_mktshr = _mkt_sample_spec.share_spec.dist_type
    _dist_firm2_pcm = _mkt_sample_spec.pcm_spec.firm2_pcm_constraint
    _hsr_filing_test_type = _mkt_sample_spec.hsr_filing_test_type

    (
        _mktshr_rng_seed_seq,
        _pcm_rng_seed_seq,
        _fcount_rng_seed_seq,
        _pr_rng_seed_seq,
    ) = parse_seed_seq_list(
        seed_seq_list, _dist_type_mktshr, _mkt_sample_spec.pr_sym_spec
    )

    _shr_sample_size = 1.0 * _mkt_sample_spec.sample_size
    # Scale up sample size to offset discards based on specified criteria
    _shr_sample_size *= _hsr_filing_test_type
    if _dist_firm2_pcm == FM2Constants.MNL:
        _shr_sample_size *= SSZConstants.MNL_DEP
    _mkt_sample_spec_here = attrs.evolve(
        _mkt_sample_spec, sample_size=int(_shr_sample_size)
    )
    del _shr_sample_size

    # Generate share data
    _mktshr_data = _gen_share_data(
        _mkt_sample_spec_here, _fcount_rng_seed_seq, _mktshr_rng_seed_seq, nthreads
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

    # Generate merging-firm price data
    _price_data = _gen_pr_data(
        _mktshr_array[:, :2], _nth_firm_share, _mkt_sample_spec_here, _pr_rng_seed_seq
    )

    _price_array, _hsr_filing_test = (
        getattr(_price_data, _f) for _f in ("price_array", "hsr_filing_test")
    )

    if _hsr_filing_test_type != SSZConstants.ONE:
        _mktshr_array = _mktshr_array[_hsr_filing_test]
        _fcounts = _fcounts[_hsr_filing_test]
        _aggregate_purchase_prob = _aggregate_purchase_prob[_hsr_filing_test]
        _nth_firm_share = _nth_firm_share[_hsr_filing_test]
        _price_array = _price_array[_hsr_filing_test]

    # Calculate diversion ratios
    _divr_array = gen_divr_array(
        _mktshr_array[:, :2],
        _mkt_sample_spec_here.recapture_rate or 0.8,
        _recapture_spec,
        _aggregate_purchase_prob,
    )

    # Generate margin data
    _pcm_data = _gen_pcm_data(
        _mktshr_array[:, :2],
        _mkt_sample_spec_here,
        _price_array,
        _aggregate_purchase_prob,
        _pcm_rng_seed_seq,
        nthreads,
    )
    _pcm_array, _mnl_test_rows = (
        getattr(_pcm_data, _f) for _f in ("pcm_array", "mnl_test_array")
    )

    _s_size = _mkt_sample_spec.sample_size  # originally-specified sample size
    if _dist_firm2_pcm == FM2Constants.MNL:
        _mktshr_array = _mktshr_array[_mnl_test_rows][:_s_size]
        _pcm_array = _pcm_array[_mnl_test_rows][:_s_size]
        _price_array = _price_array[_mnl_test_rows][:_s_size]
        _fcounts = _fcounts[_mnl_test_rows][:_s_size]
        _aggregate_purchase_prob = _aggregate_purchase_prob[_mnl_test_rows][:_s_size]
        _nth_firm_share = _nth_firm_share[_mnl_test_rows][:_s_size]
        _divr_array = _divr_array[_mnl_test_rows][:_s_size]

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
    _dist_type_mktshr: SHRConstants,
    _pr_sym_spec: PRIConstants,
    /,
) -> tuple[SeedSequence, SeedSequence, SeedSequence | None, SeedSequence | None]:
    """Initialize RNG seed sequences to ensure independence of distinct random streams."""
    _fcount_rng_seed_seq: SeedSequence | None = None
    _pr_rng_seed_seq: SeedSequence | None = None

    if _pr_sym_spec == PRIConstants.ZERO:
        _pr_rng_seed_seq = _sseq_list.pop() if _sseq_list else SeedSequence(pool_size=8)

    if _dist_type_mktshr == SHRConstants.UNI:
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

    return (
        _mktshr_rng_seed_seq,
        _pcm_rng_seed_seq,
        _fcount_rng_seed_seq,
        _pr_rng_seed_seq,
    )


def gen_divr_array(
    _frmshr_array: NDArray[np.floating[TF]],
    _r_bar: float,
    _recapture_spec: RECConstants = RECConstants.INOUT,
    _aggregate_purchase_prob: NDArray[np.floating[TF]] = EMPTY_ARRAY_DEFAULT,
    /,
) -> NDArray[np.float64]:
    """
    Given merging-firm shares and related parameters, return diverion ratios.

    If recapture is specified as "Outside-in" (RECConstants.OUTIN), then the
    choice-probability for the outside good must be supplied.

    Parameters
    ----------
    _frmshr_array
        Merging-firm shares.

    _r_bar
        If recapture is proportional or inside-out, the recapture rate
        for the firm with the smaller share.

    _aggregate_purchase_prob
        1 minus probability that the outside good is chosen; converts
        market shares to choice probabilities by multiplication.

    _recapture_spec
        Enum specifying Fixed (proportional), Inside-out, or Outside-in

    Returns
    -------
        Merging-firm diversion ratios for mergers in the sample.

    """

    _divr_array: NDArray[np.float64]
    if _recapture_spec == RECConstants.FIXED:
        _divr_array = _r_bar * _frmshr_array[:, ::-1] / (1 - _frmshr_array)

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
