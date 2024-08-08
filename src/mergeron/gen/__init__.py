"""
Defines constants, specifications (classes with attributes defining varous parameters) and
containers for industry data generation and testing.

"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import ClassVar, NamedTuple, Protocol

import numpy as np
from attrs import Attribute, cmp_using, field, frozen, validators
from numpy.random import SeedSequence

from .. import (  # noqa: TID252
    DEFAULT_REC_RATE,
    VERSION,
    ArrayBIGINT,
    ArrayBoolean,
    ArrayDouble,
    ArrayFloat,
    ArrayINT,
    RECForm,
    UPPAggrSelector,
)
from ..core.pseudorandom_numbers import DIST_PARMS_DEFAULT  # noqa: TID252

__version__ = VERSION


EMPTY_ARRAY_DEFAULT = np.zeros(2)
FCOUNT_WTS_DEFAULT = np.divide(
    (_nr := np.arange(1, 7)[::-1]), _nr.sum(), dtype=np.float64
)


class SeedSequenceData(NamedTuple):
    mktshr_rng_seed_seq: SeedSequence
    pcm_rng_seed_seq: SeedSequence
    fcount_rng_seed_seq: SeedSequence | None
    pr_rng_seed_seq: SeedSequence | None


@enum.unique
class PriceSpec(tuple[bool, str | None], enum.ReprEnum):
    """Price specification.

    Whether prices are symmetric and, if not, the direction of correlation, if any.
    """

    SYM = (True, None)
    ZERO = (False, None)
    NEG = (False, "negative share-correlation")
    POS = (False, "positive share-correlation")
    CSY = (False, "market-wide cost-symmetry")


@enum.unique
class SHRDistribution(enum.StrEnum):
    """Market share distributions."""

    UNI = "Uniform"
    R"""Uniform distribution over :math:`s_1 + s_2 \leqslant 1`"""

    DIR_FLAT = "Flat Dirichlet"
    """Shape parameter for all merging-firm-shares is unity (1)"""

    DIR_FLAT_CONSTR = "Flat Dirichlet - Constrained"
    """Impose minimum probablility weight on each firm-count

    Only firm-counts with probability weight of no less than 3%
    are included for data generation.
    """

    DIR_ASYM = "Asymmetric Dirichlet"
    """Share distribution for merging-firm shares has a higher peak share

    By default, shape parameter for merging-firm-share is 2.5, and
    1.0 for all others. Defining, :attr:`mergeron.ShareSpec.dist_parms`
    as a vector of shape parameters with length matching
    that of :attr:`mergeron.ShareSpec.dist_parms` allows flexible specification
    of Dirichlet-distributed share-data generation.
    """

    DIR_COND = "Conditional Dirichlet"
    """Shape parameters for non-merging firms is proportional

    Shape parameters for merging-firm-share are 2.0 each; and
    are equiproportional and add to 2.0 for all non-merging-firm-shares.
    """


@frozen(kw_only=False)
class ShareSpec:
    """Market share specification

    A key feature of market-share specification in this package is that
    the draws represent markets with multiple different firm-counts.
    Firm-counts are unspecified if the share distribution is
    :attr:`mergeron.SHRDistribution.UNI`, for Dirichlet-distributed market-shares,
    the default specification is that firm-counts  vary between
    2 and 7 firms with each value equally likely.

    Notes
    -----
    If :attr:`mergeron.gen.ShareSpec.dist_type`:code:` == `:attr:`mergeron.gen.SHRDistribution.UNI`,
    then it is infeasible that
    :attr:`mergeron.gen.ShareSpec.recapture_form`:code:` == `:attr:`mergeron.RECForm.OUTIN`.
    In other words, if firm-counts are unspecified, the recapture rate cannot be
    estimated using outside good choice probabilities.

    For a sample with explicit firm counts, market shares must
    be specified as having a supported Dirichlet distribution
    (see :class:`mergeron.gen.SHRDistribution`).

    """

    dist_type: SHRDistribution
    """See :class:`SHRDistribution`"""

    dist_parms: ArrayDouble | None = field(
        default=None, eq=cmp_using(eq=np.array_equal)
    )
    """Parameters for tailoring market-share distribution

    For Uniform distribution, bounds of the distribution; defaults to `(0, 1)`;
    for Dirichlet-type distributions, a vector of shape parameters of length
    no less than the length of firm-count weights below; defaults depend on
    type of Dirichlet-distribution specified.

    """
    firm_counts_weights: ArrayFloat | ArrayINT | None = field(
        default=None, eq=cmp_using(eq=np.array_equal)
    )
    """Relative or absolute frequencies of firm counts

    Given frequencies are exogenous to generated market data sample;
    for Dirichlet-type distributions, defaults to FCOUNT_WTS_DEFAULT, which specifies
    firm-counts of 2 to 6 with weights in descending order from 5 to 1.

    """

    @firm_counts_weights.validator
    def _check_fcw(_i: ShareSpec, _a: Attribute[ArrayDouble], _v: ArrayDouble) -> None:
        if _v is not None and _i.dist_type == SHRDistribution.UNI:
            raise ValueError(
                "Generated data for markets with specified firm-counts or "
                "varying firm counts are not feasible for market shares "
                "with Uniform distribution. Consider revising the "
                r"distribution type to {SHRDistribution.DIR_FLAT}, which gives "
                "uniformly distributed draws on the :math:`n+1` simplex "
                "for firm-count, :math:`n`."
            )

    recapture_form: RECForm = field(default=RECForm.INOUT)
    """See :class:`mergeron.RECForm`"""

    @recapture_form.validator
    def _check_rf(_i: ShareSpec, _a: Attribute[RECForm], _v: RECForm) -> None:
        if _v == RECForm.OUTIN and _i.dist_type == SHRDistribution.UNI:
            raise ValueError(
                "Market share specification requires estimation of recapture rate from "
                "generated data. Either delete recapture rate specification or set it to None."
            )

    recapture_rate: float | None = field(default=DEFAULT_REC_RATE)
    """A value between 0 and 1.

    :code:`None` if market share specification requires direct generation of
    outside good choice probabilities (:attr:`mergeron.RECForm.OUTIN`).

    The recapture rate is usually calibrated to the numbers-equivalent of the
    HHI threshold for the presumtion of harm from unilateral competitive effects
    in published merger guidelines. Accordingly, the recapture rate rounded to
    the nearest 5% is:

    * 0.85, **7-to-6 merger from symmetry**; US Guidelines, 1992, 2023
    * 0.80, **6-to-5 merger to symmetry**; EU Guidelines for horizontal mergers, 2004
    * 0.80, 5-to-4 merger from symmetry
    * 0.80, **5-to-4 merger to symmetry**; US Guidelines, 2010

    Highlighting indicates hypothetical mergers in the neighborhood of (the boundary of)
    the Guidelines presumption of harm. (In the EU Guidelines, concentration measures serve as
    screens for further investigation, rather than as the basis for presumptions of harm or
    presumptions no harm.)

    """

    @recapture_rate.validator
    def _check_rr(_i: ShareSpec, _a: Attribute[float], _v: float) -> None:
        if _v and not (0 < _v <= 1):
            raise ValueError("Recapture rate must lie in the interval, [0, 1).")
        elif _v is None and _i.recapture_form != RECForm.OUTIN:
            raise ValueError(
                f"Recapture specification, {_i.recapture_form!r} requires that "
                "the market sample specification inclues a recapture rate in the "
                "interval [0, 1)."
            )


@enum.unique
class PCMDistribution(enum.StrEnum):
    """Margin distributions."""

    UNI = "Uniform"
    BETA = "Beta"
    BETA_BND = "Bounded Beta"
    EMPR = "Damodaran margin data, resampled"


@enum.unique
class FM2Constraint(enum.StrEnum):
    """Firm 2 margins - derivation methods."""

    IID = "i.i.d"
    MNL = "MNL-dep"
    SYM = "symmetric"


@frozen
class PCMSpec:
    """Price-cost margin (PCM) specification

    If price-cost margins are specified as having Beta distribution,
    `dist_parms` is specified as a pair of positive, non-zero shape parameters of
    the standard Beta distribution. Specifying shape parameters :code:`np.array([1, 1])`
    is known equivalent to specifying uniform distribution over
    the interval :math:`[0, 1]`. If price-cost margins are specified as having
    Bounded-Beta distribution, `dist_parms` is specified as
    the tuple, (`mean`, `std deviation`, `min`, `max`), where `min` and `max`
    are lower- and upper-bounds respectively within the interval :math:`[0, 1]`.


    """

    dist_type: PCMDistribution = field(kw_only=False, default=PCMDistribution.UNI)
    """See :class:`PCMDistribution`"""

    dist_parms: ArrayDouble | None = field(kw_only=False, default=None)
    """Parameter specification for tailoring PCM distribution

    For Uniform distribution, bounds of the distribution; defaults to `(0, 1)`;
    for Beta distribution, shape parameters, defaults to `(1, 1)`;
    for Bounded-Beta distribution, vector of (min, max, mean, std. deviation), non-optional;
    for empirical distribution based on Damodaran margin data, optional, ignored

    """

    @dist_parms.validator
    def _check_dp(
        _i: PCMSpec, _a: Attribute[ArrayDouble | None], _v: ArrayDouble | None
    ) -> None:
        if _i.dist_type.name.startswith("BETA"):
            if _v is None:
                pass
            elif np.array_equal(_v, DIST_PARMS_DEFAULT):
                raise ValueError(
                    f"The distribution parameters, {DIST_PARMS_DEFAULT!r} "
                    "are not valid with margin distribution, {_dist_type_pcm!r}"
                )
            elif (
                _i.dist_type == PCMDistribution.BETA and len(_v) != len(("a", "b"))
            ) or (
                _i.dist_type == PCMDistribution.BETA_BND
                and len(_v) != len(("mu", "sigma", "max", "min"))
            ):
                raise ValueError(
                    f"Given number, {len(_v)} of parameters "
                    f'for PCM with distribution, "{_i.dist_type}" is incorrect.'
                )

        elif _i.dist_type == PCMDistribution.EMPR and _v is not None:
            raise ValueError(
                f"Empirical distribution does not require additional parameters; "
                f'"given value, {_v!r} is ignored."'
            )

    firm2_pcm_constraint: FM2Constraint = field(
        kw_only=False, default=FM2Constraint.IID
    )
    """See :class:`FM2Constraint`"""


@enum.unique
class SSZConstant(float, enum.ReprEnum):
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


# Validators for selected attributes of MarketSpec


@dataclass(slots=True, frozen=True)
class MarketDataSample:
    """Container for generated markets data sample."""

    frmshr_array: ArrayDouble
    """Merging-firm shares (with two merging firms)"""

    pcm_array: ArrayDouble
    """Merging-firms' prices (normalized to 1, in default specification)"""

    price_array: ArrayDouble
    """Merging-firms' price-cost margins (PCM)"""

    fcounts: ArrayBIGINT
    """Number of firms in market"""

    aggregate_purchase_prob: ArrayDouble
    """
    One (1) minus probability that the outside good is chosen

    Converts market shares to choice probabilities by multiplication.
    """

    nth_firm_share: ArrayDouble
    """Market-share of n-th firm

    Relevant for testing for draws the do or
    do not meet HSR filing thresholds.
    """

    divr_array: ArrayDouble
    """Diversion ratio between the merging firms"""

    hhi_post: ArrayDouble
    """Post-merger change in Herfindahl-Hirschmann Index (HHI)"""

    hhi_delta: ArrayDouble
    """Change in HHI from combination of merging firms"""


@dataclass(slots=True, frozen=True)
class ShareDataSample:
    """Container for generated market shares.

    Includes related measures of market structure
    and aggregate purchase probability.
    """

    mktshr_array: ArrayDouble
    """All-firm shares (with two merging firms)"""

    fcounts: ArrayBIGINT
    """All-firm-count for each draw"""

    nth_firm_share: ArrayDouble
    """Market-share of n-th firm"""

    aggregate_purchase_prob: ArrayDouble
    """Converts market shares to choice probabilities by multiplication."""


@dataclass(slots=True, frozen=True)
class PriceDataSample:
    """Container for generated price array, and related."""

    price_array: ArrayDouble
    """Merging-firms' prices"""

    hsr_filing_test: ArrayBoolean
    """Flags draws as meeting HSR filing thresholds or not"""


@dataclass(slots=True, frozen=True)
class MarginDataSample:
    """Container for generated margin array and related MNL test array."""

    pcm_array: ArrayDouble
    """Merging-firms' PCMs"""

    mnl_test_array: ArrayBoolean
    """Flags infeasible observations as False and rest as True

    Applying restrictions from Bertrand-Nash oligopoly
    with MNL demand results in draws of Firm 2 PCM falling
    outside the feabile interval,:math:`[0, 1]`, depending
    on the configuration of merging firm shares. Such draws
    are are flagged as infeasible (False)in :code:`mnl_test_array` while
    draws with PCM values within the feasible range are
    flagged as True. Used from filtering-out draws with
    infeasible PCM.
    """


@enum.unique
class INVResolution(enum.StrEnum):
    CLRN = "clearance"
    ENFT = "enforcement"
    BOTH = "both"


@frozen
class UPPTestRegime:
    """Configuration for UPP tests."""

    resolution: INVResolution = field(
        kw_only=False,
        default=INVResolution.ENFT,
        validator=validators.in_([INVResolution.CLRN, INVResolution.ENFT]),
    )
    """Whether to test clearance, enforcement, or both."""

    guppi_aggregator: UPPAggrSelector = field(
        kw_only=False, default=UPPAggrSelector.MIN
    )
    """Aggregator for GUPPI test."""

    divr_aggregator: UPPAggrSelector = field(kw_only=False, default=UPPAggrSelector.MIN)
    """Aggregator for diversion ratio test."""


@dataclass(slots=True, frozen=True)
class UPPTestsRaw:
    """Container for arrays marking test failures and successes

    A test success is a draw ("market") that meeets the
    specified test criterion, and a test failure is
    one that does not; test criteria are evaluated in
    :func:`enforcement_stats.gen_upp_arrays`.
    """

    guppi_test_simple: ArrayBoolean
    """True if GUPPI estimate meets criterion"""

    guppi_test_compound: ArrayBoolean
    """True if both GUPPI estimate and diversion ratio estimate
    meet criterion
    """

    cmcr_test: ArrayBoolean
    """True if CMCR estimate meets criterion"""

    ipr_test: ArrayBoolean
    """True if IPR (partial price-simulation) estimate meets criterion"""


@dataclass(slots=True, frozen=True)
class UPPTestsCounts:
    """Counts of markets resolved as specified

    Resolution may be either :attr:`INVResolution.ENFT`,
    :attr:`INVResolution.CLRN`, or :attr:`INVResolution.BOTH`.
    In the case of :attr:`INVResolution.BOTH`, two colums of counts
    are returned: one for each resolution.

    """

    by_firm_count: ArrayBIGINT
    by_delta: ArrayBIGINT
    by_conczone: ArrayBIGINT
    """Zones are "unoncentrated", "moderately concentrated", and "highly concentrated",
    with futher detail by HHI and ΔHHI for mergers in the "unconcentrated" and
    "moderately concentrated" zones. See
    :attr:`mergeron.gen.enforcement_stats.HMG_PRESUMPTION_ZONE_MAP` and
    :attr:`mergeron.gen.enforcement_stats.ZONE_VALS` for more detail.

    """


# https://stackoverflow.com/questions/54668000
class DataclassInstance(Protocol):
    """Generic dataclass-instance"""

    __dataclass_fields__: ClassVar
