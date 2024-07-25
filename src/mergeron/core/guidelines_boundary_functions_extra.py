"""
Specialized methods for defining and analyzing boundaries for Guidelines standards.

These methods (functions) provide rely on scipy of sympy for core computations,
and may provide improved precision than core functions, but tend to have
poor performance

"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from mpmath import mp, mpf  # type: ignore
from numpy.typing import NDArray
from scipy.spatial.distance import minkowski as distance_function  # type: ignore
from sympy import lambdify, simplify, solve, symbols  # type: ignore

from .. import VERSION  # noqa: TID252
from .guidelines_boundary_functions import (
    GuidelinesBoundary,
    _shrratio_boundary_intcpt,
    lerp,
)

__version__ = VERSION


mp.prec = 80
mp.trap_complex = True


@dataclass(slots=True, frozen=True)
class GuidelinesBoundaryCallable:
    boundary_function: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    area: float
    s_naught: float = 0


def dh_area_quad(_dh_val: float = 0.01, /, *, prec: int = 9) -> float:
    """
    Area under the ΔHHI boundary.

    When the given ΔHHI bound matches a Guidelines safeharbor,
    the area under the boundary is half the intrinsic clearance rate
    for the ΔHHI safeharbor.

    Parameters
    ----------
    _dh_val
        Merging-firms' ΔHHI bound.
    prec
        Specified precision in decimal places.

    Returns
    -------
        Area under ΔHHI boundary.

    """

    _dh_val = mpf(f"{_dh_val}")
    _s_naught = (1 - mp.sqrt(1 - 2 * _dh_val)) / 2

    return round(
        float(
            _s_naught + mp.quad(lambda x: _dh_val / (2 * x), [_s_naught, 1 - _s_naught])
        ),
        prec,
    )


def hhi_delta_boundary_qdtr(_dh_val: float = 0.01, /) -> GuidelinesBoundaryCallable:
    """
    Generate the list of share combination on the ΔHHI boundary.

    Parameters
    ----------
    _dh_val:
        Merging-firms' ΔHHI bound.

    Returns
    -------
        Callable to generate array of share-pairs, area under boundary.

    """

    _dh_val = mpf(f"{_dh_val}")

    _s_1, _s_2 = symbols("s_1, s_2", positive=True)

    _hhi_eqn = _s_2 - 0.01 / (2 * _s_1)

    _hhi_bdry = solve(_hhi_eqn, _s_2)[0]  # type: ignore
    _s_nought = float(solve(_hhi_eqn.subs({_s_2: 1 - _s_1}), _s_1)[0])  # type: ignore

    _hhi_bdry_area = 2 * (
        _s_nought
        + mp.quad(lambdify(_s_1, _hhi_bdry, "mpmath"), (_s_nought, 1 - _s_nought))
    )

    return GuidelinesBoundaryCallable(
        lambdify(_s_1, _hhi_bdry, "numpy"), _hhi_bdry_area, _s_nought
    )


def shrratio_boundary_qdtr_wtd_avg(
    _delta_star: float = 0.075,
    _r_val: float = 0.85,
    /,
    *,
    weighting: Literal["own-share", "cross-product-share"] | None = "own-share",
    recapture_form: Literal["inside-out", "proportional"] = "inside-out",
) -> GuidelinesBoundaryCallable:
    """
    Share combinations for the share-weighted average GUPPI boundary with symmetric
    merging-firm margins.

    Parameters
    ----------
    _delta_star
        corollary to GUPPI bound (:math:`\\overline{g} / (m^* \\cdot \\overline{r})`)
    _r_val
        recapture ratio
    weighting
        Whether "own-share" or "cross-product-share" (or None for simple, unweighted average)
    recapture_form
        Whether recapture-ratio is MNL-consistent ("inside-out") or has fixed
        value for both merging firms ("proportional").

    Returns
    -------
        Array of share-pairs, area under boundary.

    """

    _delta_star = mpf(f"{_delta_star}")
    _s_mid = _delta_star / (1 + _delta_star)
    _s_naught = 0

    _s_1, _s_2 = symbols("s_1:3", positive=True)

    match weighting:
        case "own-share":
            _bdry_eqn = (
                _s_1 * _s_2 / (1 - _s_1)
                + _s_2
                * _s_1
                / (
                    (1 - (_r_val * _s_2 + (1 - _r_val) * _s_1))
                    if recapture_form == "inside-out"
                    else (1 - _s_2)
                )
                - (_s_1 + _s_2) * _delta_star
            )

            _bdry_func = solve(_bdry_eqn, _s_2)[0]  # type: ignore
            _s_naught = (
                float(solve(simplify(_bdry_eqn.subs({_s_2: 1 - _s_1})), _s_1)[0])  # type: ignore
                if recapture_form == "inside-out"
                else 0
            )
            _bdry_area = float(
                2
                * (
                    _s_naught
                    + mp.quad(lambdify(_s_1, _bdry_func, "mpmath"), (_s_naught, _s_mid))
                )
                - (_s_mid**2 + _s_naught**2)
            )

        case "cross-product-share":
            mp.trap_complex = False
            _d_star = symbols("d", positive=True)
            _bdry_eqn = (
                _s_2 * _s_2 / (1 - _s_1)
                + _s_1
                * _s_1
                / (
                    (1 - (_r_val * _s_2 + (1 - _r_val) * _s_1))
                    if recapture_form == "inside-out"
                    else (1 - _s_2)
                )
                - (_s_1 + _s_2) * _d_star
            )

            _bdry_func = solve(_bdry_eqn, _s_2)[1]  # type: ignore
            _bdry_area = float(
                2
                * (
                    mp.quad(
                        lambdify(
                            _s_1, _bdry_func.subs({_d_star: _delta_star}), "mpmath"
                        ),
                        (0, _s_mid),
                    )
                ).real
                - _s_mid**2
            )

        case _:
            _bdry_eqn = (
                1 / 2 * _s_2 / (1 - _s_1)
                + 1
                / 2
                * _s_1
                / (
                    (1 - (_r_val * _s_2 + (1 - _r_val) * _s_1))
                    if recapture_form == "inside-out"
                    else (1 - _s_2)
                )
                - _delta_star
            )

            _bdry_func = solve(_bdry_eqn, _s_2)[0]  # type: ignore
            _bdry_area = float(
                2 * (mp.quad(lambdify(_s_1, _bdry_func, "mpmath"), (0, _s_mid)))
                - _s_mid**2
            )

    return GuidelinesBoundaryCallable(
        lambdify(_s_1, _bdry_func, "numpy"), _bdry_area, _s_naught
    )


def shrratio_boundary_distance(
    _delta_star: float = 0.075,
    _r_val: float = 0.85,
    /,
    *,
    agg_method: Literal["arithmetic mean", "distance"] = "arithmetic mean",
    weighting: Literal["own-share", "cross-product-share"] | None = "own-share",
    recapture_form: Literal["inside-out", "proportional"] = "inside-out",
    prec: int = 5,
) -> GuidelinesBoundary:
    """
    Share combinations for the GUPPI boundaries using various aggregators with
    symmetric merging-firm margins.

    Reimplements the arithmetic-averages and distance estimations from function,
    `shrratio_boundary_wtd_avg` but uses the Minkowski-distance function,
    `scipy.spatial.distance.minkowski` for all aggregators. This reimplementation
    is useful for testing the output of `shrratio_boundary_wtd_avg`
    but runs considerably slower.

    Parameters
    ----------
    _delta_star
        corollary to GUPPI bound (:math:`\\overline{g} / (m^* \\cdot \\overline{r})`)
    _r_val
        recapture ratio
    agg_method
        Whether "arithmetic mean" or "distance".
    weighting
        Whether "own-share" or "cross-product-share".
    recapture_form
        Whether recapture-ratio is MNL-consistent ("inside-out") or has fixed
        value for both merging firms ("proportional").
    prec
        Number of decimal places for rounding returned shares and area.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """

    _delta_star = mpf(f"{_delta_star}")
    _s_mid = _delta_star / (1 + _delta_star)

    # initial conditions
    _gbdry_points = [(_s_mid, _s_mid)]
    _s_1_pre, _s_2_pre = _s_mid, _s_mid
    _s_2_oddval, _s_2_oddsum, _s_2_evnsum = True, 0, 0

    # parameters for iteration
    _weights_base = (mpf("0.5"),) * 2
    _gbd_step_sz = mp.power(10, -prec)
    _theta = _gbd_step_sz * (10 if weighting == "cross-product-share" else 1)
    for _s_1 in mp.arange(_s_mid - _gbd_step_sz, 0, -_gbd_step_sz):
        # The wtd. avg. GUPPI is not always convex to the origin, so we
        #   increment _s_2 after each iteration in which our algorithm
        #   finds (s1, s2) on the boundary
        _s_2 = _s_2_pre * (1 + _theta)

        if (_s_1 + _s_2) > mpf("0.99875"):
            # 1: # We lose accuracy at 3-9s and up
            break

        while True:
            _de_1 = _s_2 / (1 - _s_1)
            _de_2 = (
                _s_1 / (1 - lerp(_s_1, _s_2, _r_val))
                if recapture_form == "inside-out"
                else _s_1 / (1 - _s_2)
            )

            _weights_i = (
                (
                    _w1 := mp.fdiv(
                        _s_2 if weighting == "cross-product-share" else _s_1,
                        _s_1 + _s_2,
                    ),
                    1 - _w1,
                )
                if weighting
                else _weights_base
            )

            match agg_method:
                case "arithmetic mean":
                    _delta_test = distance_function(
                        (_de_1, _de_2), (0.0, 0.0), p=1, w=_weights_i
                    )
                case "distance":
                    _delta_test = distance_function(
                        (_de_1, _de_2), (0.0, 0.0), p=2, w=_weights_i
                    )

            _test_flag, _incr_decr = (
                (_delta_test > _delta_star, -1)
                if weighting == "cross-product-share"
                else (_delta_test < _delta_star, 1)
            )

            if _test_flag:
                _s_2 += _incr_decr * _gbd_step_sz
            else:
                break

        # Build-up boundary points
        _gbdry_points.append((_s_1, _s_2))

        # Build up area terms
        _s_2_oddsum += _s_2 if _s_2_oddval else 0
        _s_2_evnsum += _s_2 if not _s_2_oddval else 0
        _s_2_oddval = not _s_2_oddval

        # Hold share points
        _s_2_pre = _s_2
        _s_1_pre = _s_1

    if _s_2_oddval:
        _s_2_evnsum -= _s_2_pre
    else:
        _s_2_oddsum -= _s_1_pre

    _s_intcpt = _shrratio_boundary_intcpt(
        _s_1_pre,
        _delta_star,
        _r_val,
        recapture_form=recapture_form,
        agg_method=agg_method,
        weighting=weighting,
    )

    if weighting == "own-share":
        _gbd_prtlarea = (
            _gbd_step_sz * (4 * _s_2_oddsum + 2 * _s_2_evnsum + _s_mid + _s_2_pre) / 3
        )
        # Area under boundary
        _gbdry_area_total = 2 * (_s_1_pre + _gbd_prtlarea) - (
            mp.power(_s_mid, "2") + mp.power(_s_1_pre, "2")
        )

    else:
        _gbd_prtlarea = (
            _gbd_step_sz * (4 * _s_2_oddsum + 2 * _s_2_evnsum + _s_mid + _s_intcpt) / 3
        )
        # Area under boundary
        _gbdry_area_total = 2 * _gbd_prtlarea - mp.power(_s_mid, "2")

    _gbdry_points = np.vstack((_gbdry_points, (mpf("0.0"), _s_intcpt))).astype(
        np.float64
    )
    # Points defining boundary to point-of-symmetry
    return GuidelinesBoundary(
        np.vstack((np.flip(_gbdry_points, 0), np.flip(_gbdry_points[1:], 1))),
        round(float(_gbdry_area_total), prec),
    )

