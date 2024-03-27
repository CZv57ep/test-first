from __future__ import annotations

from importlib.metadata import version

from attrs import Attribute, field, frozen, validators

from .. import _PKG_NAME, RECConstants, UPPAggrSelector  # noqa: TID252

__version__ = version(_PKG_NAME)


def _delta_value_validator(
    _instance: UPPBoundarySpec, _attribute: Attribute[float], _value: float, /
) -> None:
    if not 0 <= _value <= 1:
        raise ValueError(
            "Margin-adjusted benchmark share ratio must lie between 0 and 1."
        )


def _rec_spec_validator(
    _instance: UPPBoundarySpec,
    _attribute: Attribute[RECConstants],
    _value: RECConstants,
    /,
) -> None:
    if _value == RECConstants.OUTIN:
        raise ValueError(
            f"Invalid recapture specification, {_value!r}. "
            "You may consider specifying RECConstants.INOUT here, and "
            "assigning the recapture rate for the merging-firm with "
            'the smaller market-share to the attribue, "rec" of '
            "the UPPBoundarySpec object you pass."
        )
    if _value is None and _instance.agg_method != UPPAggrSelector.MAX:
        raise ValueError(
            f"Specified aggregation method, {_instance.agg_method} requires a recapture specification."
        )


@frozen
class UPPBoundarySpec:
    share_ratio: float = field(
        kw_only=False,
        default=0.075,
        validator=(validators.instance_of(float), _delta_value_validator),
    )
    rec: float = field(
        kw_only=False, default=0.80, validator=validators.instance_of(float)
    )

    agg_method: UPPAggrSelector = field(
        kw_only=True,
        default=UPPAggrSelector.MAX,
        validator=validators.instance_of(UPPAggrSelector),
    )
    recapture_form: RECConstants | None = field(
        kw_only=True,
        default=RECConstants.INOUT,
        validator=(
            validators.optional(validators.instance_of(RECConstants)),  # type: ignore
            _rec_spec_validator,
        ),
    )

    precision: int = field(
        kw_only=False, default=5, validator=validators.instance_of(int)
    )
