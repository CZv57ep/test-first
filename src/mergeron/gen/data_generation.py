"""
Functions to generate data for simulating antitrust and merger analysis.

"""

import enum
from importlib import metadata
from pathlib import Path
from typing import Literal, NamedTuple, TypeAlias

import attrs
import numpy as np
from numpy.random import SeedSequence
from numpy.typing import NDArray

from mergeron.core.damodaran_margin_data import resample_mgn_data
from mergeron.core.pseudorandom_numbers import (
    DIST_PARMS_DEFAULT,
    MultithreadedRNG,
    prng,
)

__version__ = metadata.version(Path(__file__).parents[1].stem)

EMPTY_ARRAY_DEFAULT = np.zeros(2)
FCOUNT_WTS_DEFAULT = (_nr := np.arange(1, 6)[::-1]) / _nr.sum()


@enum.unique
class SHRConstants(enum.StrEnum):
    """Market share distributions."""

    UNI = "Uniform"
    DIR_FLAT = "Flat Dirichlet"
    DIR_FLAT_CONSTR = "Flat Dirichlet - Constrained"
    DIR_ASYM = "Asymmetric Dirichlet"
    DIR_COND = "Conditional Dirichlet"


@enum.unique
class PRIConstants(tuple[bool, str | None], enum.ReprEnum):
    """Price specification.

    Whether prices are symmetric and, if not, the direction of correlation, if any.
    """

    SYM = (True, None)
    ZERO = (False, None)
    NEG = (False, "negative")
    POS = (False, "positive")


@enum.unique
class PCMConstants(enum.StrEnum):
    """Margin distributions."""

    UNI = "Uniform"
    BETA = "Beta"
    BETA_BND = "Bounded Beta"
    EMPR = "AD, NYU (N-F)"


@enum.unique
class FM2Constants(enum.StrEnum):
    """Firm 2 margins - derivation methods."""

    IID = "i.i.d"
    MNL = "MNL-dep"
    SYM = "symmetric"


@enum.unique
class RECConstants(enum.StrEnum):
    """Recapture rate - derivation methods."""

    INOUT = "inside-out"
    OUTIN = "outside-in"
    FIXED = "proportional"


@enum.unique
class SSZConstants(float, enum.ReprEnum):
    """
    Scale factors to offset sample size reduction.

    Sample size reduction occurs when imposing a HSR filing test
    or equilibrium condition under MNL demand.
    """

    HSR_NTH = 1.666667
    """
    For HSR filing requirement.

    When filing requirement is assumed met if maximum merging-firm shares exceeds
    ten (10) times the n-th firm's share and minimum merging-firm share is
    no less than n-th firm's share. To assure that the number of draws available
    after applying the given restriction, the initial number of draws is larger than
    the sample size by the given scale factor.
    """

    HSR_TEN = 1.234567
    """
    For alternative HSR filing requirement,

    When filing requirement is assumed met if merging-firm shares exceed 10:1 ratio
    to each other.
    """

    MNL_DEP = 1.25
    """
    For restricted PCM's.

    When merging firm's PCMs are constrained for consistency with f.o.c.s from
    profit maximization under Nash-Bertrand oligopoly with MNL demand.
    """

    ONE = 1.00
    """When initial set of draws is not restricted in any way."""


ShareSpec: TypeAlias = tuple[
    SHRConstants, RECConstants, NDArray[np.int64 | np.float64] | None
]

PCMSpec: TypeAlias = tuple[PCMConstants, FM2Constants, NDArray[np.float64]]


@attrs.define(frozen=True)  # auto_attribs=False,
class MarketSampleSpec:
    """Parameter specification for market data generation."""

    sample_size: int = 10**6  # attrs.field(default=)
    recapture_rate: float = 0.80  # attrs.field(default=)
    pr_sym_spec: PRIConstants = PRIConstants.SYM  # attrs.field(default=)
    share_spec: ShareSpec = attrs.field(
        kw_only=True, default=(SHRConstants.UNI, RECConstants.INOUT, None)
    )
    pcm_spec: PCMSpec = attrs.field(
        kw_only=True, default=(PCMConstants.UNI, FM2Constants.IID, DIST_PARMS_DEFAULT)
    )
    hsr_filing_test_type: SSZConstants = attrs.field(
        kw_only=True, default=SSZConstants.ONE
    )


class MarketSample(NamedTuple):
    """Container for generated market data sample."""

    frmshr_array: NDArray[np.float64]
    pcm_array: NDArray[np.float64]
    price_array: NDArray[np.float64]
    fcounts: NDArray[np.int64]
    choice_prob_outgd: NDArray[np.float64]
    nth_firm_share: NDArray[np.float64]
    divr_array: NDArray[np.float64]
    hhi_post: NDArray[np.float64]
    hhi_delta: NDArray[np.float64]


class ShareDataSample(NamedTuple):
    """Container forestimated market shares, and related."""

    mktshr_array: NDArray[np.float64]
    fcounts: NDArray[np.int64]
    nth_firm_share: NDArray[np.float64]
    choice_prob_outgd: NDArray[np.float64]


class PriceDataSample(NamedTuple):
    """Container for generated price array, and related."""

    price_array: NDArray[np.float64]
    hsr_filing_test: NDArray[np.bool_]


class DivrDataSample(NamedTuple):
    """Container for estimated diversion ratio array, and related."""

    divr_array: NDArray[np.float64]
    choice_prob_outgd: NDArray[np.float64]


class MarginDataSample(NamedTuple):
    """Container for generated margin array and related MNL test array."""

    pcm_array: NDArray[np.float64]
    mnl_test_array: NDArray[np.bool_]


def gen_market_sample(
    _mkt_sample_spec: MarketSampleSpec,
    /,
    *,
    seed_seq_list: list[SeedSequence] | None = None,
    nthreads: int = 16,
) -> MarketSample:
    """
    Generate share, diversion ratio, price, and margin data based on supplied parameters

    Diversion ratios generated assuming share-proportionality, unless
    `recapture_spec`="proportional", in which case both firms' recapture rate
    is set to `r_bar`.

    If price-cost margins are specified as having Beta distribution,
    `pcm_dist_parms` can be specified as an interval on [0, 1], or
    as a tuple, (`mean`, `std deviation`, `min`, `max`), np.where `min` and `max`
    are 0 and 1, respectively, or may bind PCMs to a sub-interval of [0, 1].

    The tuple of SeedSequences, if specified, is parsed in the following order
    for generating the relevant random variates:
    1.) quantity shares
    2.) price-cost margins
    3.) firm-counts, from [2, 2 + len(firm_counts_prob_weights)] where relevant
    4.) prices, if pr_sym_spec[1] and not pr_sym_spec[2]

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

    _dist_type_mktshr, _recapture_spec, _ = _mkt_sample_spec.share_spec
    _, _dist_firm2_pcm, _ = _mkt_sample_spec.pcm_spec
    _hsr_filing_test_type = _mkt_sample_spec.hsr_filing_test_type

    if _recapture_spec == RECConstants.FIXED and _dist_firm2_pcm == FM2Constants.MNL:
        raise ValueError(
            "{} {} {}".format(
                f'Specification of "recapture_spec", "{_recapture_spec}"',
                "requires Firm 2 margin be distributed, ",
                f'"{FM2Constants.IID}" or "{FM2Constants.SYM}".',
            )
        )

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

    # Generate merging-firm price data
    _price_data = _gen_pr_ratio(
        _mktshr_data.mktshr_array[:, :2],
        _mktshr_data.nth_firm_share,
        _mkt_sample_spec_here,
        _pr_rng_seed_seq,
    )

    (_mktshr_array, _fcounts, _choice_prob_outgd, _nth_firm_share, _price_array) = (
        _f[_price_data.hsr_filing_test]  # type: ignore
        if _hsr_filing_test_type != SSZConstants.ONE
        else _f
        for _f in (
            _mktshr_data.mktshr_array,
            _mktshr_data.fcounts,
            _mktshr_data.choice_prob_outgd,
            _mktshr_data.nth_firm_share,
            _price_data.price_array,
        )
    )
    del _mktshr_data, _price_data

    # Calculate diversion ratios
    if _recapture_spec == RECConstants.OUTIN:
        #   Outside-in calibration only valid for Dir-distributed shares
        if not _dist_type_mktshr.name.startswith("DIR_"):
            raise ValueError(
                "{} {} {}".format(
                    f"Value of _recapture_spec, {_recapture_spec}",
                    " is invalid. Must be a Dirichlet-type ",
                    f"distribution among, {SHRConstants.__dict__['_members_']!r}",
                )
            )
    _divr_array, _choice_prob_outgd = gen_divr_array(
        _mktshr_array[:, :2],
        _mkt_sample_spec_here.recapture_rate,
        _recapture_spec,
        _choice_prob_outgd,
    )

    # Generate margin data
    _pcm_array, _mnl_test_rows = _gen_pcm_data(
        _mktshr_array[:, :2],
        _mkt_sample_spec_here,
        _price_array,
        _choice_prob_outgd,
        _pcm_rng_seed_seq,
        nthreads,
    )

    _s_size = _mkt_sample_spec.sample_size  # originally-specified sample size
    (
        _mktshr_array,
        _pcm_array,
        _price_array,
        _fcounts,
        _choice_prob_outgd,
        _nth_firm_share,
        _divr_array,
    ) = (
        _f[_mnl_test_rows][:_s_size]
        if _dist_firm2_pcm == FM2Constants.MNL
        else _f[:_s_size]
        for _f in (
            _mktshr_array,
            _pcm_array,
            _price_array,
            _fcounts,
            _choice_prob_outgd,
            _nth_firm_share,
            _divr_array,
        )
    )
    del _mnl_test_rows, _s_size

    _frmshr_array = _mktshr_array[:, :2]
    _hhi_delta = np.einsum("ij,ij->i", _frmshr_array, _frmshr_array[:, ::-1])[:, None]

    _hhi_post = (
        _hhi_delta + np.einsum("ij,ij->i", _mktshr_array, _mktshr_array)[:, None]
    )

    return MarketSample(
        _frmshr_array,
        _pcm_array,
        _price_array,
        _fcounts,
        _choice_prob_outgd,
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
    """Set up RNG seed sequences, to ensure independence of random streams.

    We go through this step to ensure that seeded random generation of
    market data is consistent with provided parameters
    """
    _fcount_rng_seed_seq: SeedSequence | None = None
    _pr_rng_seed_seq: SeedSequence | None = None

    if not _pr_sym_spec or _pr_sym_spec == PRIConstants.SYM:
        _pr_rng_seed_seq = None
    elif not _sseq_list:
        _pr_rng_seed_seq = SeedSequence(pool_size=8)
    else:
        _pr_rng_seed_seq = _sseq_list.pop()

    if _sseq_list:
        if _dist_type_mktshr == SHRConstants.UNI:
            _fcount_rng_seed_seq = None
            _mktshr_rng_seed_seq, _pcm_rng_seed_seq = _sseq_list[:2]
        else:
            (_mktshr_rng_seed_seq, _pcm_rng_seed_seq, _fcount_rng_seed_seq) = (
                _sseq_list[:3]
            )
    elif _dist_type_mktshr == SHRConstants.UNI:
        _fcount_rng_seed_seq = None
        _mktshr_rng_seed_seq, _pcm_rng_seed_seq = (
            SeedSequence(pool_size=8) for _ in range(2)
        )
    else:
        _mktshr_rng_seed_seq, _pcm_rng_seed_seq, _fcount_rng_seed_seq = (
            SeedSequence(pool_size=8) for _ in range(3)
        )

    return (
        _mktshr_rng_seed_seq,
        _pcm_rng_seed_seq,
        _fcount_rng_seed_seq,
        _pr_rng_seed_seq,
    )


def _gen_share_data(
    _mkt_sample_spec: MarketSampleSpec,
    _fcount_rng_seed_seq: SeedSequence | None,
    _mktshr_rng_seed_seq: SeedSequence,
    _nthreads: int = 16,
    /,
) -> ShareDataSample:
    """Helper function for generating share data.

    Parameters
    ----------
    _mkt_sample_spec
        Class specifying parameters for share-, price-, and margin-data generation
    _fcount_rng_seed_seq
        Seed sequence for assuring independent and, optionally, redundant streams
    _mktshr_rng_seed_seq
        Seed sequence for assuring independent and, optionally, redundant streams
    _nthreads
        Must be specified for generating repeatable random streams

    Returns
    -------
        Arrays representing shares, diversion ratios, etc. structured as a :ShareDataSample:

    """

    _dist_type_mktshr, _recapture_spec, _firm_count_prob_wts_raw = (
        _mkt_sample_spec.share_spec
    )

    _ssz = _mkt_sample_spec.sample_size

    # If recapture_spec == "inside-out", further calculations downstream
    if _dist_type_mktshr == SHRConstants.UNI:
        if _recapture_spec == RECConstants.OUTIN:
            raise ValueError(
                "{} {} {}".format(
                    'When, mktshr_dist_type is "Uniform",',
                    'recapture_spec must not be "outside-in".',
                    "Correct the function call and re-run your code.",
                )
            )
        _mkt_share_sample = gen_market_shares_uniform(
            _ssz, _mktshr_rng_seed_seq, _nthreads
        )

    elif _dist_type_mktshr.name.startswith("DIR_"):
        _firm_count_prob_wts = (
            None
            if _firm_count_prob_wts_raw is None
            else np.array(_firm_count_prob_wts_raw, dtype=np.float64)
        )
        _mkt_share_sample = gen_market_shares_dirichlet_multisample(
            _ssz,
            _dist_type_mktshr,
            _recapture_spec,
            _firm_count_prob_wts,
            _fcount_rng_seed_seq,
            _mktshr_rng_seed_seq,
            _nthreads,
        )

    else:
        raise ValueError(
            f'Unexpected type, "{_dist_type_mktshr}" for share distribution.'
        )

    return _mkt_share_sample


def gen_market_shares_uniform(
    _s_size: int = 10**6,
    _mktshr_rng_seed_seq: SeedSequence | None = None,
    _nthreads: int = 16,
    /,
) -> ShareDataSample:
    """Generate merging-firm shares from Uniform distribution on the 3-D simplex.

    Parameters
    ----------
    _s_size
        size of sample to be drawn
    _mktshr_rng_seed_seq
        seed for rng, so results can be made replicable
    _nthreads
        number of threads for random number generation

    Returns
    -------
        market shares and other market statistics for each draw (market)

    """

    _frmshr_array = np.empty((_s_size, 2), dtype=np.float64)
    _mrng = MultithreadedRNG(
        _frmshr_array,
        dist_type="Uniform",
        seed_sequence=_mktshr_rng_seed_seq,
        nthreads=_nthreads,
    )
    _mrng.fill()
    # Convert draws on U[0, 1] to Uniformly-distributed draws on simplex, s_1 + s_2 < 1
    _frmshr_array = np.sort(_frmshr_array, axis=1)
    _frmshr_array = np.column_stack((
        _frmshr_array[:, 0],
        _frmshr_array[:, 1] - _frmshr_array[:, 0],
    ))

    # Keep only share combinations representing feasible mergers
    _frmshr_array = _frmshr_array[_frmshr_array.min(axis=1) > 0]

    # Let a third column have values of "np.nan", so HHI calculations return "np.nan"
    _mktshr_array = np.pad(
        _frmshr_array, ((0, 0), (0, 1)), "constant", constant_values=np.nan
    )

    return ShareDataSample(
        _mktshr_array, *np.split(np.nan * np.empty((_s_size, 3)), 3, axis=1)
    )


def gen_market_shares_dirichlet_multisample(
    _s_size: int = 10**6,
    _dist_type_dir: SHRConstants = SHRConstants.DIR_FLAT,
    _recapture_spec: RECConstants = RECConstants.INOUT,
    _firm_count_wts: NDArray[np.floating] = FCOUNT_WTS_DEFAULT,
    _fcount_rng_seed_seq: SeedSequence | None = None,
    _mktshr_rng_seed_seq: SeedSequence | None = None,
    _nthreads: int = 16,
    /,
) -> ShareDataSample:
    """Dirichlet-distributed shares with multiple firm-counts.

    Firm-counts may be specified as having Uniform distribution over the range
    of firm counts, or a set of probability weights may be specified. In the
    latter case the proportion of draws for each firm-count matches the
    specified probability weight.

    Parameters
    ----------
    _s_size
        sample size to be drawn
    _firm_count_wts
        firm count weights array for sample to be drawn
    _dist_type_dir
        Whether Dirichlet is Flat or Asymmetric
    _recapture_spec
        r_1 = r_2 if "proportional", otherwise MNL-consistent
    _fcount_rng_seed_seq
        seed firm count rng, for replicable results
    _mktshr_rng_seed_seq
        seed market share rng, for replicable results
    _nthreads
        number of threads for parallelized random number generation

    Returns
    -------
        array of market shares and other market statistics

    """

    _min_choice_wt = 0.03 if _dist_type_dir == SHRConstants.DIR_FLAT_CONSTR else 0.00
    _fcount_keys, _choice_wts = zip(
        *(
            _f
            for _f in zip(
                2 + np.arange(len(_firm_count_wts)),
                _firm_count_wts / _firm_count_wts.sum(),
                strict=True,
            )
            if _f[1] > _min_choice_wt
        )
    )
    _choice_wts = _choice_wts / sum(_choice_wts)

    _fc_max = _fcount_keys[-1]
    _dir_alphas_full = [1.0] * _fc_max
    if _dist_type_dir == SHRConstants.DIR_ASYM:
        _dir_alphas_full = [2.0] * 6 + [1.5] * 5 + [1.25] * min(7, _fc_max)

    if _dist_type_dir == SHRConstants.DIR_COND:

        def _gen_dir_alphas(_fcv: int) -> NDArray[np.float64]:
            _dat = [2.5] * 2
            if _fcv > len(_dat):
                _dat += [1.0 / (_fcv - 2)] * (_fcv - 2)
            return np.array(_dat, dtype=np.float64)

    else:

        def _gen_dir_alphas(_fcv: int) -> NDArray[np.float64]:
            return np.array(_dir_alphas_full[:_fcv], dtype=np.float64)

    _fcounts = prng(_fcount_rng_seed_seq).choice(
        _fcount_keys, size=(_s_size, 1), p=_choice_wts
    )

    _mktshr_seed_seq_ch = (
        _mktshr_rng_seed_seq.spawn(len(_fcount_keys))
        if isinstance(_mktshr_rng_seed_seq, SeedSequence)
        else SeedSequence(pool_size=8).spawn(len(_fcounts))
    )

    _choice_prob_outgd, _nth_firm_share = (np.empty((_s_size, 1)) for _ in range(2))
    _mktshr_array = np.empty((_s_size, _fc_max), dtype=np.float64)
    for _f_val, _f_sseq in zip(_fcount_keys, _mktshr_seed_seq_ch, strict=True):
        _fcounts_match_rows = np.where(_fcounts == _f_val)[0]
        _dir_alphas_test = _gen_dir_alphas(_f_val)

        try:
            _mktshr_sample_f = gen_market_shares_dirichlet(
                _dir_alphas_test,
                len(_fcounts_match_rows),
                _recapture_spec,
                _f_sseq,
                _nthreads,
            )
        except ValueError as _err:
            print(_f_val, len(_fcounts_match_rows))
            raise _err

        # Push data for present sample to parent
        _mktshr_array[_fcounts_match_rows] = np.pad(
            _mktshr_sample_f.mktshr_array,
            ((0, 0), (0, _fc_max - _mktshr_sample_f.mktshr_array.shape[1])),
            "constant",
        )
        _choice_prob_outgd[_fcounts_match_rows] = _mktshr_sample_f.choice_prob_outgd
        _nth_firm_share[_fcounts_match_rows] = _mktshr_sample_f.nth_firm_share

    if (_iss := np.round(np.einsum("ij->", _mktshr_array))) != _s_size or _iss != len(
        _mktshr_array
    ):
        raise ValueError(
            "DATA GENERATION ERROR: {} {} {}".format(
                "Generation of sample shares is inconsistent:",
                "array of drawn shares must some to the number of draws",
                "i.e., the sample size, which condition is not met.",
            )
        )

    return ShareDataSample(_mktshr_array, _fcounts, _nth_firm_share, _choice_prob_outgd)


def gen_market_shares_dirichlet(
    _dir_alphas: NDArray[np.floating],
    _s_size: int = 10**6,
    _recapture_spec: RECConstants = RECConstants.INOUT,
    _mktshr_rng_seed_seq: SeedSequence | None = None,
    _nthreads: int = 16,
    /,
) -> ShareDataSample:
    """Dirichlet-distributed shares with fixed firm-count.

    Parameters
    ----------
    _dir_alphas
        Shape parameters for Dirichlet distribution
    _s_size
        sample size to be drawn
    _recapture_spec
        r_1 = r_2 if RECConstants.FIXED, otherwise MNL-consistent. If
        RECConstants.OUTIN; the number of columns in the output share array
        is len(_dir_alphas) - 1.
    _mktshr_rng_seed_seq
        seed market share rng, for replicable results
    _nthreads
        number of threads for parallelized random number generation

    Returns
    -------
        array of market shares and other market statistics

    """

    if not isinstance(_dir_alphas, np.ndarray):
        _dir_alphas = np.array(_dir_alphas)

    if _recapture_spec == RECConstants.OUTIN:
        _dir_alphas = np.concatenate((_dir_alphas, _dir_alphas[-1:]))

    _mktshr_seed_seq_ch = (
        _mktshr_rng_seed_seq
        if isinstance(_mktshr_rng_seed_seq, SeedSequence)
        else SeedSequence(pool_size=8)
    )

    _mktshr_array = np.empty((_s_size, len(_dir_alphas)))
    _mrng = MultithreadedRNG(
        _mktshr_array,
        dist_type="Dirichlet",
        dist_parms=_dir_alphas,
        seed_sequence=_mktshr_seed_seq_ch,
        nthreads=_nthreads,
    )
    _mrng.fill()

    if (_iss := np.round(np.einsum("ij->", _mktshr_array))) != _s_size or _iss != len(
        _mktshr_array
    ):
        print(_dir_alphas, _iss, repr(_s_size), len(_mktshr_array))
        print(repr(_mktshr_array[-10:, :]))
        raise ValueError(
            "DATA GENERATION ERROR: {} {} {}".format(
                "Generation of sample shares is inconsistent:",
                "array of drawn shares must sum to the number of draws",
                "i.e., the sample size, which condition is not met.",
            )
        )

    # If recapture_spec == 'inside_out', further calculations downstream
    _choice_prob_outgd = np.nan * np.empty((_s_size, 1))
    if _recapture_spec == RECConstants.OUTIN:
        _choice_prob_outgd = _mktshr_array[:, [-1]]
        _mktshr_array = _mktshr_array[:, :-1] / (1 - _choice_prob_outgd)

    return ShareDataSample(
        _mktshr_array,
        (_mktshr_array.shape[-1] * np.ones((_s_size, 1))).astype(np.int64),
        _mktshr_array[:, [-1]],
        _choice_prob_outgd,
    )


def _gen_pr_ratio(
    _frmshr_array: NDArray[np.floating],
    _nth_firm_share: NDArray[np.floating],
    _mkt_sample_spec: MarketSampleSpec,
    _seed_seq: SeedSequence | None = None,
    /,
) -> PriceDataSample:
    _ssz = len(_frmshr_array)

    _hsr_filing_test_type = _mkt_sample_spec.hsr_filing_test_type

    _price_array, _price_ratio_array, _hsr_filing_test = (
        np.ones_like(_frmshr_array),
        np.empty_like(_frmshr_array),
        np.empty(_ssz, dtype=bool),
    )

    _pr_max_ratio = 5.0
    if _mkt_sample_spec.pr_sym_spec == PRIConstants.SYM:
        _nth_firm_price = np.ones((_ssz, 1))
    elif _mkt_sample_spec.pr_sym_spec == PRIConstants.POS:
        _price_array, _nth_firm_price = (
            np.ceil(_p * _pr_max_ratio) for _p in (_frmshr_array, _nth_firm_share)
        )
    elif _mkt_sample_spec.pr_sym_spec == PRIConstants.NEG:
        _price_array, _nth_firm_price = (
            np.ceil((1 - _p) * _pr_max_ratio) for _p in (_frmshr_array, _nth_firm_share)
        )
    elif _mkt_sample_spec.pr_sym_spec == PRIConstants.ZERO:
        _price_array_gen = prng(_seed_seq).choice(
            1 + np.arange(_pr_max_ratio), size=(len(_frmshr_array), 3)
        )
        _price_array = _price_array_gen[:, :2]
        _nth_firm_price = _price_array_gen[:, [2]]
        # del _price_array_gen
    else:
        raise ValueError(
            f"Condition regarding price symmetry"
            f' "{_mkt_sample_spec.pr_sym_spec.value}" is invalid.'
        )
    # del _pr_max_ratio

    _price_ratio_array = _price_array / _price_array[:, ::-1]
    _rev_array = _price_array * _frmshr_array
    _nth_firm_rev = _nth_firm_price * _nth_firm_share

    # Although `_test_rev_ratio_inv` is not fixed at 10%,
    # the ratio has not changed since inception of the HSR filing test,
    # so we treat it as a constant of merger policy.
    _test_rev_ratio, _test_rev_ratio_inv = 10, 1 / 10

    if _hsr_filing_test_type == SSZConstants.HSR_TEN:
        # See, https://www.ftc.gov/enforcement/premerger-notification-program/
        #   -> Procedures For Submitting Post-Consummation Filings
        #    -> Key Elements to Determine Whether a Post Consummation Filing is Required
        #           under heading, "Historical Thresholds"
        # Revenue ratio has been 10-to-1 since inception
        # Thus, a simple form of the HSR filing test would impose a 10-to-1
        # ratio restriction on the merging firms' revenues
        _rev_ratio = (_rev_array.min(axis=1) / _rev_array.max(axis=1)).round(4)
        _hsr_filing_test = _rev_ratio >= _test_rev_ratio_inv
        # del _rev_array, _rev_ratio
    elif _hsr_filing_test_type == SSZConstants.HSR_NTH:
        # To get around the 10-to-1 ratio restriction, specify that the nth firm
        # matches the smaller firm in the size test; then if the smaller merging firm
        # matches the n-th firm in size, and the larger merging firm has at least
        # 10 times the size of the nth firm, the size test is considered met.
        # Alternatively, if the smaller merging firm has 10% or greater share,
        # the value of transaction test is considered met.
        _rev_ratio_to_nth = np.round(np.sort(_rev_array, axis=1) / _nth_firm_rev, 4)
        _hsr_filing_test = (
            np.einsum(
                "ij->i", 1 * (_rev_ratio_to_nth > [1, _test_rev_ratio]), dtype=np.int64
            )
            == _rev_ratio_to_nth.shape[1]
        ) | (_frmshr_array.min(axis=1) >= _test_rev_ratio_inv)

        # del _nth_firm_rev, _rev_ratio_to_nth
    else:
        # Otherwise, all draws meet the filing test
        _hsr_filing_test = np.ones(_ssz, dtype=bool)

    return PriceDataSample(_price_array, _hsr_filing_test)


def gen_divr_array(
    _frmshr_array: NDArray[np.floating],
    _r_bar: float,
    _recapture_spec: RECConstants = RECConstants.INOUT,
    _choice_prob_outgd: NDArray[np.floating] = EMPTY_ARRAY_DEFAULT,
    /,
) -> DivrDataSample:
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

    _choice_prob_outgd
        Probability that the outside good is chosen.

    _recapture_spec
        Enum specifying Fixed (proportional), Inside-out, or Outside-in

    Returns
    -------
        Merging-firm diversion ratios for mergers in the sample.

    """
    if np.array_equal(_choice_prob_outgd, EMPTY_ARRAY_DEFAULT):
        _choice_prob_outgd = np.empty((len(_frmshr_array), 1))
    _one_minus_choice_prob_outgd = np.empty_like(_choice_prob_outgd)

    if _recapture_spec == RECConstants.FIXED:
        _divr_array = _r_bar * _frmshr_array[:, ::-1] / (1 - _frmshr_array)

    else:
        _one_minus_choice_prob_outgd = (
            1 - _choice_prob_outgd
            if _recapture_spec == RECConstants.OUTIN
            else (
                _r_bar / (1 - (1 - _r_bar) * _frmshr_array.min(axis=1, keepdims=True))
            )
        )
        _purchprob_array = _one_minus_choice_prob_outgd * _frmshr_array
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

    return DivrDataSample(
        _divr_array,
        _choice_prob_outgd
        if _recapture_spec in (RECConstants.FIXED, RECConstants.OUTIN)
        else 1 - _one_minus_choice_prob_outgd,
    )


def _gen_pcm_data(
    _frmshr_array: NDArray[np.floating],
    _mkt_sample_spec: MarketSampleSpec,
    _price_array: NDArray[np.floating],
    _choice_prob_outgd: NDArray[np.floating],
    _pcm_rng_seed_seq: SeedSequence,
    _nthreads: int = 16,
    /,
) -> MarginDataSample:
    _, _recapture_spec, _ = _mkt_sample_spec.share_spec
    _dist_type_pcm, _dist_firm2_pcm, _dist_parms_pcm = _mkt_sample_spec.pcm_spec

    _pcm_array = np.empty((len(_frmshr_array), 2), dtype=np.float64)
    _mnl_test_array = np.empty((len(_frmshr_array), 2), dtype=int)

    _beta_min, _beta_max = [None] * 2  # placeholder
    if _dist_type_pcm == PCMConstants.EMPR:
        _pcm_array = resample_mgn_data(
            _pcm_array.shape, seed_sequence=_pcm_rng_seed_seq
        )
    else:
        _dist_type: Literal["Uniform", "Beta"] = _dist_type_pcm.value  # type: ignore
        _dist_parms = _dist_parms_pcm
        if _dist_type_pcm.name.startswith("BETA"):
            if (
                _dist_type_pcm == PCMConstants.BETA
                and len(_dist_parms_pcm) != len(("max", "min"))
            ) or (
                _dist_type_pcm == PCMConstants.BETA_BND
                and len(_dist_parms_pcm) != len(("mu", "sigma", "max", "min"))
            ):
                raise ValueError(
                    f"Given number, {len(_dist_parms_pcm)} of parameters "
                    f'for PCM with distribution, "{_dist_type_pcm}" is incorrect.'
                )
            if _dist_type_pcm == PCMConstants.BETA_BND:  # Bounded beta
                _dist_parms = _beta_located_bound(_dist_parms_pcm)
                _dist_type = "Beta"

        _pcm_rng = MultithreadedRNG(
            _pcm_array,
            dist_type=_dist_type,
            dist_parms=_dist_parms,
            seed_sequence=_pcm_rng_seed_seq,
            nthreads=_nthreads,
        )
        _pcm_rng.fill()
        # del _pcm_rng

    if _mkt_sample_spec.pcm_spec[0] == PCMConstants.BETA_BND:
        _beta_min, _beta_max = _mkt_sample_spec.pcm_spec[2][2:]
        _pcm_array = (_beta_max - _beta_min) * _pcm_array + _beta_min
        # del _beta_min, _beta_max

    if _dist_firm2_pcm == FM2Constants.MNL and _recapture_spec != RECConstants.FIXED:
        # Impose FOCs from profit-maximization with MNL demand
        _purchprob_array = (1 - _choice_prob_outgd) * _frmshr_array

        _pcm_array[:, [1]] = np.divide(
            np.einsum(
                "ij,ij,ij->ij",
                _price_array[:, [0]],
                _pcm_array[:, [0]],
                1 - _purchprob_array[:, [0]],
            ),
            np.einsum("ij,ij->ij", _price_array[:, [1]], 1 - _purchprob_array[:, [1]]),
        )

        _mnl_test_array = _pcm_array[:, 1].__ge__(0) & _pcm_array[:, 1].__le__(1)
    else:
        _mnl_test_array = np.ones(len(_pcm_array), dtype=bool)
        if _dist_firm2_pcm == FM2Constants.SYM:
            _pcm_array[:, [1]] = _pcm_array[:, [0]]

    return MarginDataSample(_pcm_array, _mnl_test_array)


def _beta_located(
    _mu: float | NDArray[np.float64], _sigma: float | NDArray[np.float64], /
) -> NDArray[np.float64]:
    """
    Given mean and stddev, return shape parameters for corresponding Beta distribution

    Solve the first two moments of the standard Beta to get the shape parameters. [1]_

    Parameters
    ----------
    _mu
        mean
    _sigma
        standardd deviation

    Returns
    -------
        shape parameters for Beta distribution

    References
    ----------
    .. [1] NIST. https://www.itl.nist.gov/div898/handbook/eda/section3/eda366h.htm

    """
    _mul = (_mu - _mu**2 - _sigma**2) / _sigma**2
    return np.array([_mu * _mul, (1 - _mu) * _mul], dtype=np.float64)


def beta_located_bound(_dist_parms: NDArray[np.floating], /) -> NDArray[np.float64]:
    R"""
    Return shape parameters for a non-standard beta, given the mean, stddev, range


    Recover the r.v.s as :math:`\min + (\max - \min) \cdot \symup{Β}(a, b)`, with `a` and `b`
    calculated from the specified mean (:math:`\mu`) and variance (:math:`\sigma`). [7]_

    Parameters
    ----------
    _dist_parms_pcm
        vector of :math:`\mu`, :math:`\sigma`, :math:`\mathtt{\min}`, and :math:`\mathtt{\max}` values

    Returns
    -------
        shape parameters for Beta distribution

    Notes
    -----
    For example, ``beta_located_bound(np.array([0.5, 0.2887, 0.0, 1.0]))``.

    References
    ----------
    .. [7] NIST. https://www.itl.nist.gov/div898/handbook/eda/section3/eda366h.htm
    """

    _bmu, _bsigma, _bmin, _bmax = _dist_parms
    return _beta_located((_bmu - _bmin) / (_bmax - _bmin), _bsigma / (_bmax - _bmin))
