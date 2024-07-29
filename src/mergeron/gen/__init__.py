"""
Defines constants and containers for industry data generation and testing

"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from attrs import Attribute, cmp_using, define, field, frozen, validators
from numpy.typing import NDArray

from .. import VERSION, RECConstants, UPPAggrSelector  # noqa: TID252
from ..core.pseudorandom_numbers import DIST_PARMS_DEFAULT  # noqa: TID252

__version__ = VERSION


EMPTY_ARRAY_DEFAULT = np.zeros(2)
FCOUNT_WTS_DEFAULT = np.divide(
    (_nr := np.arange(1, 7)[::-1]), _nr.sum(), dtype=np.float64
)


@enum.unique
class PriceConstants(tuple[bool, str | None], enum.ReprEnum):
    """Price specification.

    Whether prices are symmetric and, if not, the direction of correlation, if any.
    """

    SYM = (True, None)
    ZERO = (False, None)
    NEG = (False, "negative share-correlation")
    POS = (False, "positive share-correlation")
    CSY = (False, "market-wide cost-symmetry")


@enum.unique
class SHRConstants(enum.StrEnum):
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


@frozen
class ShareSpec:
    """Market share specification

    A key feature of market-share specification in this package is that
    the draws represent markets with multiple different firm-counts.
    Firm-counts are unspecified if the share distribution is
    :attr:`mergeron.SHRConstants.UNI`, for Dirichlet-distributed market-shares,
    the default specification is that firm-counts  vary between
    2 and 7 firms with each value equally likely.

    Notes
    -----
    If :attr:`mergeron.gen.ShareSpec.dist_type`:code:` == `:attr:`mergeron.gen.SHRConstants.UNI`,
    then it is infeasible that
    :attr:`mergeron.gen.ShareSpec.recapture_form`:code:` == `:attr:`mergeron.RECConstants.OUTIN`.
    In other words, if firm-counts are unspecified, the recapture rate cannot be
    estimated using outside good choice probabilities.

    For a sample with explicit firm counts, market shares must
    be specified as having a supported Dirichlet distribution
    (see :class:`mergeron.gen.SHRConstants`).

    """

    recapture_form: RECConstants
    """See :class:`mergeron.RECConstants`"""

    recapture_rate: float | None
    """A value between 0 and 1, typically 0.8.

    :code:`None` if market share specification requires direct generation of
    outside good choice probabilities (:attr:`mergeron.RECConstants.OUTIN`).

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

    dist_type: SHRConstants
    """See :class:`SHRConstants`"""

    dist_parms: NDArray[np.float64] | None = field(
        default=None, eq=cmp_using(eq=np.array_equal)
    )
    """Parameters for tailoring market-share distribution

    For Uniform distribution, bounds of the distribution; defaults to `(0, 1)`;
    for Dirichlet-type distributions, a vector of shape parameters of length
    no less than the length of firm-count weights below; defaults depend on
    type of Dirichlet-distribution specified.

    """
    firm_counts_weights: NDArray[np.float64 | np.int64] | None = field(
        default=None, eq=cmp_using(eq=np.array_equal)
    )
    """Relative or absolute frequencies of firm counts


    Given frequencies are exogenous to generated market data sample;
    for Dirichlet-type distributions, defaults to FCOUNT_WTS_DEFAULT, which specifies
    firm-counts of 2 to 6 with weights in descending order from 5 to 1.

    """


@enum.unique
class PCMConstants(enum.StrEnum):
    """Margin distributions."""

    UNI = "Uniform"
    BETA = "Beta"
    BETA_BND = "Bounded Beta"
    EMPR = "Damodaran margin data, resampled"


@enum.unique
class FM2Constants(enum.StrEnum):
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

    firm2_pcm_constraint: FM2Constants
    """See :class:`FM2Constants`"""

    dist_type: PCMConstants
    """See :class:`PCMConstants`"""

    dist_parms: NDArray[np.float64] | None
    """Parameter specification for tailoring PCM distribution

    For Uniform distribution, bounds of the distribution; defaults to `(0, 1)`;
    for Beta distribution, shape parameters, defaults to `(1, 1)`;
    for Bounded-Beta distribution, vector of (min, max, mean, std. deviation), non-optional;
    for empirical distribution based on Damodaran margin data, optional, ignored

    """


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


# Validators for selected attributes of MarketSpec
def _share_spec_validator(
    _instance: MarketSpec, _attribute: Attribute[ShareSpec], _value: ShareSpec, /
) -> None:
    _r_bar = _value.recapture_rate
    if _r_bar and not (0 < _r_bar <= 1):
        raise ValueError("Recapture rate must lie in the interval, [0, 1).")

    elif _r_bar and _value.recapture_form == RECConstants.OUTIN:
        raise ValueError(
            "Market share specification requires estimation of recapture rate from "
            "generated data. Either delete recapture rate specification or set it to None."
        )

    if _value.dist_type == SHRConstants.UNI:
        if _value.recapture_form == RECConstants.OUTIN:
            raise ValueError(
                f"Invalid recapture specification, {_value.recapture_form!r} "
                "for market share specification with Uniform distribution. "
                "Redefine the market-sample specification, modifying the ."
                "market-share specification or the recapture specification."
            )
        elif _value.firm_counts_weights is not None:
            raise ValueError(
                "Generated data for markets with specified firm-counts or "
                "varying firm counts are not feasible for market shares "
                "with Uniform distribution. Consider revising the "
                r"distribution type to {SHRConstants.DIR_FLAT}, which gives "
                "uniformly distributed draws on the :math:`n+1` simplex "
                "for firm-count, :math:`n`."
            )
    elif _value.recapture_form != RECConstants.OUTIN and (
        _r_bar is None or not isinstance(_r_bar, float)
    ):
        raise ValueError(
            f"Recapture specification, {_value.recapture_form!r} requires that "
            "the market sample specification inclues a recapture rate."
        )


def _pcm_spec_validator(
    _instance: MarketSpec, _attribute: Attribute[PCMSpec], _value: PCMSpec, /
) -> None:
    if (
        _instance.share_spec.recapture_form == RECConstants.FIXED
        and _value.firm2_pcm_constraint == FM2Constants.MNL
    ):
        raise ValueError(
            "{} {} {}".format(
                f'Specification of "recapture_form", "{_instance.share_spec.recapture_form}"',
                "requires Firm 2 margin must have property, ",
                f'"{FM2Constants.IID}" or "{FM2Constants.SYM}".',
            )
        )
    elif _value.dist_type.name.startswith("BETA"):
        if _value.dist_parms is None:
            pass
        elif np.array_equal(_value.dist_parms, DIST_PARMS_DEFAULT):
            raise ValueError(
                f"The distribution parameters, {DIST_PARMS_DEFAULT!r} "
                "are not valid with margin distribution, {_dist_type_pcm!r}"
            )
        elif (
            _value.dist_type == PCMConstants.BETA
            and len(_value.dist_parms) != len(("max", "min"))
        ) or (
            _value.dist_type == PCMConstants.BETA_BND
            and len(_value.dist_parms) != len(("mu", "sigma", "max", "min"))
        ):
            raise ValueError(
                f"Given number, {len(_value.dist_parms)} of parameters "
                f'for PCM with distribution, "{_value.dist_type}" is incorrect.'
            )


@define(slots=False)
class MarketSpec:
    """Parameter specification for market data generation."""

    share_spec: ShareSpec = field(
        kw_only=True,
        default=ShareSpec(RECConstants.INOUT, 0.85, SHRConstants.UNI, None, None),
        validator=[validators.instance_of(ShareSpec), _share_spec_validator],
    )
    """Market-share specification, see :class:`ShareSpec`"""

    pcm_spec: PCMSpec = field(
        kw_only=True,
        default=PCMSpec(FM2Constants.IID, PCMConstants.UNI, None),
        validator=[validators.instance_of(PCMSpec), _pcm_spec_validator],
    )
    """Margin specification, see :class:`PCMSpec`"""

    price_spec: PriceConstants = field(
        kw_only=True,
        default=PriceConstants.SYM,
        validator=validators.instance_of(PriceConstants),
    )
    """Price specification, see :class:`PriceConstants`"""

    hsr_filing_test_type: SSZConstants = field(
        kw_only=True,
        default=SSZConstants.ONE,
        validator=validators.instance_of(SSZConstants),
    )
    """Method for modeling HSR filing threholds, see :class:`SSZConstants`"""


@dataclass(slots=True, frozen=True)
class MarketDataSample:
    """Container for generated markets data sample."""

    frmshr_array: NDArray[np.float64]
    """Merging-firm shares (with two merging firms)"""

    pcm_array: NDArray[np.float64]
    """Merging-firms' prices (normalized to 1, in default specification)"""

    price_array: NDArray[np.float64]
    """Merging-firms' price-cost margins (PCM)"""

    fcounts: NDArray[np.int64]
    """Number of firms in market"""

    aggregate_purchase_prob: NDArray[np.float64]
    """
    One (1) minus probability that the outside good is chosen

    Converts market shares to choice probabilities by multiplication.
    """

    nth_firm_share: NDArray[np.float64]
    """Market-share of n-th firm

    Relevant for testing for draws the do or
    do not meet HSR filing thresholds.
    """

    divr_array: NDArray[np.float64]
    """Diversion ratio between the merging firms"""

    hhi_post: NDArray[np.float64]
    """Post-merger change in Herfindahl-Hirschmann Index (HHI)"""

    hhi_delta: NDArray[np.float64]
    """Change in HHI from combination of merging firms"""


@dataclass(slots=True, frozen=True)
class ShareDataSample:
    """Container for generated market shares.

    Includes related measures of market structure
    and aggregate purchase probability.
    """

    mktshr_array: NDArray[np.float64]
    """All-firm shares (with two merging firms)"""

    fcounts: NDArray[np.int64]
    """All-firm-count for each draw"""

    nth_firm_share: NDArray[np.float64]
    """Market-share of n-th firm"""

    aggregate_purchase_prob: NDArray[np.float64]
    """Converts market shares to choice probabilities by multiplication."""


@dataclass(slots=True, frozen=True)
class PriceDataSample:
    """Container for generated price array, and related."""

    price_array: NDArray[np.float64]
    """Merging-firms' prices"""

    hsr_filing_test: NDArray[np.bool_]
    """Flags draws as meeting HSR filing thresholds or not"""


@dataclass(slots=True, frozen=True)
class MarginDataSample:
    """Container for generated margin array and related MNL test array."""

    pcm_array: NDArray[np.float64]
    """Merging-firms' PCMs"""

    mnl_test_array: NDArray[np.bool_]
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
    resolution: INVResolution = field(
        default=INVResolution.ENFT, validator=validators.instance_of(INVResolution)
    )
    guppi_aggregator: UPPAggrSelector = field(
        default=UPPAggrSelector.MIN, validator=validators.instance_of(UPPAggrSelector)
    )
    divr_aggregator: UPPAggrSelector | None = field(
        default=UPPAggrSelector.MIN,
        validator=validators.instance_of((UPPAggrSelector, type(None))),
    )


@dataclass(slots=True, frozen=True)
class UPPTestsRaw:
    """Container for arrays marking test failures and successes

    A test success is a draw ("market") that meeets the
    specified test criterion, and a test failure is
    one that does not; test criteria are evaluated in
    :func:`enforcement_stats.gen_upp_arrays`.
    """

    guppi_test_simple: NDArray[np.bool_]
    """True if GUPPI estimate meets criterion"""

    guppi_test_compound: NDArray[np.bool_]
    """True if both GUPPI estimate and diversion ratio estimate
    meet criterion
    """

    cmcr_test: NDArray[np.bool_]
    """True if CMCR estimate meets criterion"""

    ipr_test: NDArray[np.bool_]
    """True if IPR (partial price-simulation) estimate meets criterion"""


@dataclass(slots=True, frozen=True)
class UPPTestsCounts:
    """Counts of markets resolved as specified

    Resolution may be either :attr:`INVResolution.ENFT` or :attr:`INVResolution.CLRN`.
    """

    by_firm_count: NDArray[np.int64]
    by_delta: NDArray[np.int64]
    by_conczone: NDArray[np.int64]
    """Zones are "unoncentrated", "moderately concentrated", and "highly concentrated"
    """


# https://stackoverflow.com/questions/54668000
class DataclassInstance(Protocol):
    """Generic dataclass-instance"""

    __dataclass_fields__: ClassVar
