"""
Defines constants and containers for industry data generation and testing

"""

from __future__ import annotations

from importlib.metadata import version

from .. import _PKG_NAME  # noqa: TID252

__version__ = version(_PKG_NAME)


import enum
from dataclasses import dataclass
from typing import ClassVar, Protocol, TypeVar

import attrs
import numpy as np
from numpy.typing import NBitBase, NDArray

from ..core.pseudorandom_numbers import DIST_PARMS_DEFAULT  # noqa: TID252

EMPTY_ARRAY_DEFAULT = np.zeros(2)
FCOUNT_WTS_DEFAULT = ((_nr := np.arange(1, 6)[::-1]) / _nr.sum()).astype(np.float64)

TF = TypeVar("TF", bound=NBitBase)
TI = TypeVar("TI", bound=NBitBase)


# https://stackoverflow.com/questions/54668000
class DataclassInstance(Protocol):
    """For type-hinting dataclass objects"""

    __dataclass_asdict__: ClassVar
    __dataclass_fields__: ClassVar


@enum.unique
class PRIConstants(tuple[bool, str | None], enum.ReprEnum):
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
    """Uniform distribution over the 3-simplex"""

    DIR_FLAT = "Flat Dirichlet"
    """Shape parameter for all merging-firm-shares is unity (1)"""

    DIR_FLAT_CONSTR = "Flat Dirichlet - Constrained"
    """Impose minimum probablility weight on each firm-count

    Only firm-counts with probability weight of no less than 3%
    are included for data generation.
    """

    DIR_ASYM = "Asymmetric Dirichlet"
    """Share distribution for merging-firm shares has a higher peak share

    Shape parameter for merging-firm-share is 2.5, and 1.0 for all others.
    """

    DIR_COND = "Conditional Dirichlet"
    """Shape parameters for non-merging firms is proportional

    Shape parameters for merging-firm-share are 2.0 each; and
    are equiproportional and add to 2.0 for all non-merging-firm-shares.
    """


@enum.unique
class RECConstants(enum.StrEnum):
    """Recapture rate - derivation methods."""

    INOUT = "inside-out"
    OUTIN = "outside-in"
    FIXED = "proportional"


@attrs.define(slots=True, frozen=True)
class ShareSpec:
    """Market share specification

    Notes
    -----
    If recapture is determined "outside-in", market shares cannot have
    Uniform distribution.

    If sample with varying firm counts is required, market shares must
    be specified as having a supported Dirichlet distribution.

    """

    recapture_spec: RECConstants
    """see RECConstants"""

    dist_type: SHRConstants
    """see SHRConstants"""

    dist_parms: NDArray[np.float64] | None
    """Parameters for tailoring market-share distribution

    For Uniform distribution, bounds of the distribution; defaults to `(0, 1)`;
    for Beta distribution, shape parameters, defaults to `(1, 1)`;
    for Bounded-Beta distribution, vector of (min, max, mean, std. deviation), non-optional;
    for Dirichlet-type distributions, a vector of shape parameters of length
    no less than the length of firm-count weights below; defaults depend on
    type of Dirichlet-distribution specified.

    """
    firm_counts_weights: NDArray[np.float64 | np.int64] | None
    """relative or absolute frequencies of firm counts


    Given frequencies are exogenous to generated market data sample;
    defaults to FCOUNT_WTS_DEFAULT, which specifies firm-counts of 2 to 6
    with weights in descending order from 5 to 1."""


@enum.unique
class PCMConstants(enum.StrEnum):
    """Margin distributions."""

    UNI = "Uniform"
    BETA = "Beta"
    BETA_BND = "Bounded Beta"
    EMPR = "Damodaran margin data"


@enum.unique
class FM2Constants(enum.StrEnum):
    """Firm 2 margins - derivation methods."""

    IID = "i.i.d"
    MNL = "MNL-dep"
    SYM = "symmetric"


@attrs.define(slots=True, frozen=True)
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

    dist_type: PCMConstants
    """See PCMConstants"""

    firm2_pcm_constraint: FM2Constants
    """See FM2Constants"""

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


# Validators for selected attributes of MarketSampleSpec
def _sample_size_validator(
    _object: MarketSampleSpec, _attribute: attrs.Attribute, _value: int, /
) -> None:
    if _value < 10**6 or (_value % 10**5 != 0):
        raise ValueError(
            f"Sample size must be a multiple of {10** 4:,d} and not less than {10** 6:,d}."
        )


def _recapture_rate_validator(
    _object: MarketSampleSpec, _attribute: attrs.Attribute, _value: float | None, /
) -> None:
    # if _value < 10**6 or (np.log10(_value) % 1 != 0):
    if _value and not (0 < _value <= 1):
        raise ValueError("Recapture rate must lie in the interval, [0, 1).")

    if _value and _object.share_spec.recapture_spec == RECConstants.OUTIN:
        raise ValueError(
            "Market share specification requires estimation of recapture rate from "
            "generated data. Either delete recapture rate specification or set it to None."
        )


def _share_spec_validator(
    _instance: MarketSampleSpec, _attribute: attrs.Attribute, _value: ShareSpec, /
) -> None:
    _r_bar = _instance.recapture_rate
    if _value.dist_type == SHRConstants.UNI:
        if _value.recapture_spec == RECConstants.OUTIN:
            raise ValueError(
                f"Invalid recapture specification, {_value.recapture_spec!r} "
                "for market share specification with Uniform distribution. "
                "Redefine the market-sample specification, modifying the ."
                "market-share specification or the recapture specification."
            )
        elif _value.firm_counts_weights is not None:
            raise ValueError(
                "Generated data for markets with specified firm-counts or "
                "varying firm counts are not feasible with market shares "
                "with Uniform distribution. Consider revising the "
                r"distribution type to {SHRConstants.DIR_FLAT}, which gives "
                "uniformly distributed draws on the :math:`n+1` simplex "
                "for firm-count, :math:`n`."
            )
        #   Outside-in calibration only valid for Dir-distributed shares
    elif _value.recapture_spec != RECConstants.OUTIN and (
        _r_bar is None or not isinstance(_r_bar, float)
    ):
        raise ValueError(
            f"Recapture specification, {_value.recapture_spec!r} requires that "
            "the market sample specification inclues a recapture rate."
        )


def _pcm_spec_validator(
    _instance: MarketSampleSpec, _attribute: attrs.Attribute, _value: PCMSpec, /
) -> None:
    if (
        _instance.share_spec.recapture_spec == RECConstants.FIXED
        and _value.firm2_pcm_constraint == FM2Constants.MNL
    ):
        raise ValueError(
            "{} {} {}".format(
                f'Specification of "recapture_spec", "{_instance.share_spec.recapture_spec}"',
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


@attrs.define(slots=True, frozen=True)
class MarketSampleSpec:
    """Parameter specification for market data generation."""

    sample_size: int = attrs.field(
        default=10**6,
        validator=(attrs.validators.instance_of(int), _sample_size_validator),
    )
    """sample size generated"""

    recapture_rate: float | None = attrs.field(
        default=None, validator=attrs.validators.instance_of(float | None)
    )
    """market recapture rate

    Is None if market share specification requires generation of
    outside good choice probabilities (RECConstants.OUTIN).
    """

    pr_sym_spec: PRIConstants = attrs.field(  # type: ignore
        kw_only=True,
        default=PRIConstants.SYM,
        validator=attrs.validators.instance_of(PRIConstants),  # type: ignore
    )
    """Price specification, see PRIConstants"""

    share_spec: ShareSpec = attrs.field(
        kw_only=True,
        default=ShareSpec(RECConstants.INOUT, SHRConstants.UNI, None, None),
        validator=[attrs.validators.instance_of(ShareSpec), _share_spec_validator],
    )
    """See definition of ShareSpec"""

    pcm_spec: PCMSpec = attrs.field(
        kw_only=True,
        default=PCMSpec(PCMConstants.UNI, FM2Constants.IID, None),
        validator=[attrs.validators.instance_of(PCMSpec), _pcm_spec_validator],
    )
    """See definition of PCMSpec"""

    hsr_filing_test_type: SSZConstants = attrs.field(  # type: ignore
        kw_only=True,
        default=SSZConstants.ONE,
        validator=attrs.validators.instance_of(SSZConstants),  # type: ignore
    )
    """Method for modeling HSR filing threholds, see SSZConstants"""


@dataclass(slots=True, frozen=True)
class MarketDataSample:
    """Container for generated markets data sample."""

    frmshr_array: NDArray[np.floating]
    """Merging-firm shares (with two merging firms)"""

    pcm_array: NDArray[np.floating]
    """Merging-firms' prices (normalized to 1, in default specification)"""

    price_array: NDArray[np.floating]
    """Merging-firms' price-cost margins (PCM)"""

    fcounts: NDArray[np.integer]
    """Number of firms in market"""

    ratio_choice_prob_to_mktshr: NDArray[np.floating]
    """
    One (1) minus probability that the outside good is chosen

    Converts market shares to choice probabilities by multiplication.
    """

    nth_firm_share: NDArray[np.floating]
    """Market-share of n-th firm

    Relevant for testing for draws the do or
    do not meet HSR filing thresholds.
    """

    divr_array: NDArray[np.floating]
    """Diversion ratio between the merging firms"""

    hhi_post: NDArray[np.floating]
    """Post-merger change in Herfindahl-Hirschmann Index (HHI)"""

    hhi_delta: NDArray[np.floating]
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

    price_array: NDArray[np.floating]
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


@dataclass(slots=True, frozen=True)
class UPPTestsRaw:
    guppi_test_simple: NDArray[np.bool_]
    guppi_test_compound: NDArray[np.bool_]
    cmcr_test: NDArray[np.bool_]
    ipr_test: NDArray[np.bool_]


@dataclass(slots=True, frozen=True)
class UPPTestsCounts:
    by_firm_count: NDArray[np.int64]
    by_delta: NDArray[np.int64]
    by_conczone: NDArray[np.int64]
