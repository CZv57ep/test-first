import decimal
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np
from mpmath import mp, mpf  # type: ignore
from numpy.typing import NDArray

from .. import VERSION  # noqa: TID252

__version__ = VERSION

mp.prec = 80
mp.trap_complex = True


class ShareRatioBoundaryKeywords(TypedDict, total=False):
    """Keyword arguments for functions generating share ratio boundaries."""

    recapture_form: Literal["inside-out", "proportional"]
    prec: int
    agg_method: Literal["arithmetic mean", "geometric mean", "distance"]
    weighting: Literal["own-share", "cross-product-share", None]


@dataclass(frozen=True)
class GuidelinesBoundary:
    """Output of a Guidelines boundary function."""

    coordinates: NDArray[np.float64]
    """Market-share pairs as Cartesian coordinates of points on the boundary."""

    area: float
    """Area under the boundary."""


def dh_area(_dh_val: float = 0.01, /, *, prec: int = 9) -> float:
    R"""
    Area under the ΔHHI boundary.

    When the given ΔHHI bound matches a Guidelines standard,
    the area under the boundary is half the intrinsic clearance rate
    for the ΔHHI safeharbor.

    Notes
    -----
    To derive the knots, :math:`(s^0_1, s^1_1), (s^1_1, s^0_1)`
    of the ΔHHI boundary, i.e., the points where it intersects
    the merger-to-monopoly boundary, solve

    .. math::

        2 s1 s_2 &= ΔHHI\\
        s_1 + s_2 &= 1

    Parameters
    ----------
    _dh_val
        Change in concentration.
    prec
        Specified precision in decimal places.

    Returns
    -------
        Area under ΔHHI boundary.

    """

    _dh_val = mpf(f"{_dh_val}")
    _s_naught = (1 - mp.sqrt(1 - 2 * _dh_val)) / 2

    return round(
        float(_s_naught + (_dh_val / 2) * (mp.ln(1 - _s_naught) - mp.ln(_s_naught))),
        prec,
    )


def hhi_delta_boundary(
    _dh_val: float = 0.01, /, *, prec: int = 5
) -> GuidelinesBoundary:
    """
    Generate the list of share combination on the ΔHHI boundary.

    Parameters
    ----------
    _dh_val:
        Merging-firms' ΔHHI bound.
    prec
        Number of decimal places for rounding reported shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """

    _dh_val = mpf(f"{_dh_val}")
    _s_naught = 1 / 2 * (1 - mp.sqrt(1 - 2 * _dh_val))
    _s_mid = mp.sqrt(_dh_val / 2)

    _dh_step_sz = mp.power(10, -6)
    _s_1 = np.array(mp.arange(_s_mid, _s_naught - mp.eps, -_dh_step_sz))
    _s_2 = _dh_val / (2 * _s_1)

    # Boundary points
    _dh_half = np.vstack((
        np.column_stack((_s_1, _s_2)),
        np.array([(mpf("0.0"), mpf("1.0"))]),
    ))
    _dh_bdry_pts = np.vstack((np.flip(_dh_half, 0), np.flip(_dh_half[1:], 1)))

    _s_1_pts, _s_2_pts = np.split(_dh_bdry_pts, 2, axis=1)
    return GuidelinesBoundary(
        np.column_stack((
            np.array(_s_1_pts, np.float64),
            np.array(_s_2_pts, np.float64),
        )),
        dh_area(_dh_val, prec=prec),
    )


def hhi_pre_contrib_boundary(
    _hhi_contrib: float = 0.03125, /, *, prec: int = 5
) -> GuidelinesBoundary:
    """
    Share combinations on the premerger HHI contribution boundary.

    Parameters
    ----------
    _hhi_contrib:
        Merging-firms' pre-merger HHI contribution bound.
    prec
        Number of decimal places for rounding reported shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """
    _hhi_contrib = mpf(f"{_hhi_contrib}")
    _s_mid = mp.sqrt(_hhi_contrib / 2)

    _bdry_step_sz = mp.power(10, -prec)
    # Range-limit is 0 less a step, which is -1 * step-size
    _s_1 = np.array(mp.arange(_s_mid, -_bdry_step_sz, -_bdry_step_sz), np.float64)
    _s_2 = np.sqrt(_hhi_contrib - _s_1**2).astype(np.float64)
    _bdry_pts_mid = np.column_stack((_s_1, _s_2))
    return GuidelinesBoundary(
        np.vstack((np.flip(_bdry_pts_mid, 0), np.flip(_bdry_pts_mid[1:], 1))),
        round(float(mp.pi * _hhi_contrib / 4), prec),
    )


def combined_share_boundary(
    _s_intcpt: float = 0.0625, /, *, prec: int = 10
) -> GuidelinesBoundary:
    """
    Share combinations on the merging-firms' combined share boundary.

    Assumes symmetric merging-firm margins. The combined-share is
    congruent to the post-merger HHI contribution boundary, as the
    post-merger HHI bound is the square of the combined-share bound.

    Parameters
    ----------
    _s_intcpt:
        Merging-firms' combined share.
    prec
        Number of decimal places for rounding reported shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """
    _s_intcpt = mpf(f"{_s_intcpt}")
    _s_mid = _s_intcpt / 2

    _s1_pts = (0, _s_mid, _s_intcpt)
    return GuidelinesBoundary(
        np.column_stack((
            np.array(_s1_pts, np.float64),
            np.array(_s1_pts[::-1], np.float64),
        )),
        round(float(_s_intcpt * _s_mid), prec),
    )


def hhi_post_contrib_boundary(
    _hhi_contrib: float = 0.800, /, *, prec: int = 10
) -> GuidelinesBoundary:
    """
    Share combinations on the postmerger HHI contribution boundary.

    The post-merger HHI contribution boundary is identical to the
    combined-share boundary.

    Parameters
    ----------
    _hhi_contrib:
        Merging-firms' pre-merger HHI contribution bound.
    prec
        Number of decimal places for rounding reported shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """
    return combined_share_boundary(np.sqrt(_hhi_contrib), prec=prec)


def shrratio_boundary_wtd_avg(
    _delta_star: float = 0.075,
    _r_val: float = 0.85,
    /,
    *,
    agg_method: Literal[
        "arithmetic mean", "geometric mean", "distance"
    ] = "arithmetic mean",
    weighting: Literal["own-share", "cross-product-share", None] = "own-share",
    recapture_form: Literal["inside-out", "proportional"] = "inside-out",
    prec: int = 5,
) -> GuidelinesBoundary:
    """
    Share combinations on the share-weighted average diversion ratio boundary.

    Parameters
    ----------
    _delta_star
        Share ratio (:math:`\\overline{d} / \\overline{r}`)
    _r_val
        recapture ratio
    agg_method
        Whether "arithmetic mean", "geometric mean", or "distance".
    weighting
        Whether "own-share" or "cross-product-share"  (or None for simple, unweighted average).
    recapture_form
        Whether recapture-ratio is MNL-consistent ("inside-out") or has fixed
        value for both merging firms ("proportional").
    prec
        Number of decimal places for rounding returned shares and area.

    Returns
    -------
        Array of share-pairs, area under boundary.

    Notes
    -----
    An analytical expression for the share-weighted arithmetic mean boundary
    is derived and plotted from y-intercept to the ray of symmetry as follows::

        from sympy import plot as symplot, solve, symbols
        s_1, s_2 = symbols("s_1 s_2", positive=True)

        g_val, r_val, m_val = 0.06, 0.80, 0.30
        delta_star = g_val / (r_val * m_val)

        # recapture_form == "inside-out"
        oswag = solve(
            s_1 * s_2 / (1 - s_1)
            + s_2 * s_1 / (1 - (r_val * s_2 + (1 - r_val) * s_1))
            - (s_1 + s_2) * delta_star,
            s_2
        )[0]
        symplot(
            oswag,
            (s_1, 0., d_hat / (1 + d_hat)),
            ylabel=s_2
        )

        cpswag = solve(
            s_2 * s_2 / (1 - s_1)
            + s_1 * s_1 / (1 - (r_val * s_2 + (1 - r_val) * s_1))
            - (s_1 + s_2) * delta_star,
            s_2
        )[1]
        symplot(
            cpwag,
            (s_1, 0., d_hat / (1 + d_hat)),
            ylabel=s_2
        )

        # recapture_form == "proportional"
        oswag = solve(
            s_1 * s_2 / (1 - s_1)
            + s_2 * s_1 / (1 - s_2)
            - (s_1 + s_2) * delta_star,
             s_2
        )[0]
        symplot(
            oswag,
            (s_1, 0., d_hat / (1 + d_hat)),
            ylabel=s_2
        )

        cpswag = solve(
            s_2 * s_2 / (1 - s_1)
            + s_1 * s_1 / (1 - s_2)
            - (s_1 + s_2) * delta_star,
             s_2
        )[1]
        symplot(
            cpswag,
            (s_1, 0.0, d_hat / (1 + d_hat)),
            ylabel=s_2
        )


    """

    _delta_star = mpf(f"{_delta_star}")
    _s_mid = _delta_star / (1 + _delta_star)

    # initial conditions
    _gbdry_points = [(_s_mid, _s_mid)]
    _s_1_pre, _s_2_pre = _s_mid, _s_mid
    _s_2_oddval, _s_2_oddsum, _s_2_evnsum = True, 0.0, 0.0

    # parameters for iteration
    _gbd_step_sz = mp.power(10, -prec)
    _theta = _gbd_step_sz * (10 if weighting == "cross-product-share" else 1)
    for _s_1 in mp.arange(_s_mid - _gbd_step_sz, 0, -_gbd_step_sz):
        # The wtd. avg. GUPPI is not always convex to the origin, so we
        #   increment _s_2 after each iteration in which our algorithm
        #   finds (s1, s2) on the boundary
        _s_2 = _s_2_pre * (1 + _theta)

        if (_s_1 + _s_2) > mpf("0.99875"):
            # Loss of accuracy at 3-9s and up
            break

        while True:
            _de_1 = _s_2 / (1 - _s_1)
            _de_2 = (
                _s_1 / (1 - lerp(_s_1, _s_2, _r_val))
                if recapture_form == "inside-out"
                else _s_1 / (1 - _s_2)
            )

            _r = (
                mp.fdiv(
                    _s_1 if weighting == "cross-product-share" else _s_2, _s_1 + _s_2
                )
                if weighting
                else 0.5
            )

            match agg_method:
                case "geometric":
                    _delta_test = mp.expm1(lerp(mp.log1p(_de_1), mp.log1p(_de_2), _r))
                case "distance":
                    _delta_test = mp.sqrt(lerp(_de_1**2, _de_2**2, _r))
                case _:
                    _delta_test = lerp(_de_1, _de_2, _r)

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
        _gbdry_area_total = float(
            2 * (_s_1_pre + _gbd_prtlarea)
            - (mp.power(_s_mid, "2") + mp.power(_s_1_pre, "2"))
        )

    else:
        _gbd_prtlarea = (
            _gbd_step_sz * (4 * _s_2_oddsum + 2 * _s_2_evnsum + _s_mid + _s_intcpt) / 3
        )
        # Area under boundary
        _gbdry_area_total = float(2 * _gbd_prtlarea - mp.power(_s_mid, "2"))

    _gbdry_points = np.vstack((_gbdry_points, (mpf("0.0"), _s_intcpt))).astype(
        np.float64
    )

    # Points defining boundary to point-of-symmetry
    return GuidelinesBoundary(
        np.vstack((np.flip(_gbdry_points, 0), np.flip(_gbdry_points[1:], 1))),
        round(float(_gbdry_area_total), prec),
    )


def shrratio_boundary_xact_avg(
    _delta_star: float = 0.075,
    _r_val: float = 0.85,
    /,
    *,
    recapture_form: Literal["inside-out", "proportional"] = "inside-out",
    prec: int = 5,
) -> GuidelinesBoundary:
    """
    Share combinations for the simple average GUPPI boundary with symmetric
    merging-firm margins.

    Notes
    -----
    An analytical expression for the exact average boundary is derived
    and plotted from the y-intercept to the ray of symmetry as follows::

        from sympy import latex, plot as symplot, solve, symbols

        s_1, s_2 = symbols("s_1 s_2")

        g_val, r_val, m_val = 0.06, 0.80, 0.30
        d_hat = g_val / (r_val * m_val)

        # recapture_form = "inside-out"
        sag = solve(
            (s_2 / (1 - s_1))
            + (s_1 / (1 - (r_val * s_2 + (1 - r_val) * s_1)))
            - 2 * d_hat,
            s_2
        )[0]
        symplot(
            sag,
            (s_1, 0., d_hat / (1 + d_hat)),
            ylabel=s_2
        )

        # recapture_form = "proportional"
        sag = solve((s_2/(1 - s_1)) + (s_1/(1 - s_2)) - 2 * d_hat, s_2)[0]
        symplot(
            sag,
            (s_1, 0., d_hat / (1 + d_hat)),
            ylabel=s_2
        )

    Parameters
    ----------
    _delta_star
        Share ratio (:math:`\\overline{d} / \\overline{r}`).
    _r_val
        Recapture ratio
    recapture_form
        Whether recapture-ratio is MNL-consistent ("inside-out") or has fixed
        value for both merging firms ("proportional").
    prec
        Number of decimal places for rounding returned shares.

    Returns
    -------
        Array of share-pairs, area under boundary, area under boundary.

    """

    _delta_star = mpf(f"{_delta_star}")
    _s_mid = _delta_star / (1 + _delta_star)
    _gbd_step_sz = mp.power(10, -prec)

    _gbdry_points_start = np.array([(_s_mid, _s_mid)])
    _s_1 = np.array(mp.arange(_s_mid - _gbd_step_sz, 0, -_gbd_step_sz), np.float64)
    if recapture_form == "inside-out":
        _s_intcpt = mp.fdiv(
            mp.fsub(
                2 * _delta_star * _r_val + 1, mp.fabs(2 * _delta_star * _r_val - 1)
            ),
            2 * mpf(f"{_r_val}"),
        )
        _nr_t1 = 1 + 2 * _delta_star * _r_val * (1 - _s_1) - _s_1 * (1 - _r_val)

        _nr_sqrt_mdr = 4 * _delta_star * _r_val
        _nr_sqrt_mdr2 = _nr_sqrt_mdr * _r_val
        _nr_sqrt_md2r2 = _nr_sqrt_mdr2 * _delta_star

        _nr_sqrt_t1 = _nr_sqrt_md2r2 * (_s_1**2 - 2 * _s_1 + 1)
        _nr_sqrt_t2 = _nr_sqrt_mdr2 * _s_1 * (_s_1 - 1)
        _nr_sqrt_t3 = _nr_sqrt_mdr * (2 * _s_1 - _s_1**2 - 1)
        _nr_sqrt_t4 = (_s_1**2) * (_r_val**2 - 6 * _r_val + 1)
        _nr_sqrt_t5 = _s_1 * (6 * _r_val - 2) + 1

        _nr_t2_mdr = _nr_sqrt_t1 + _nr_sqrt_t2 + _nr_sqrt_t3 + _nr_sqrt_t4 + _nr_sqrt_t5

        # Alternative grouping of terms in np.sqrt
        _nr_sqrt_s1sq = (_s_1**2) * (
            _nr_sqrt_md2r2 + _nr_sqrt_mdr2 - _nr_sqrt_mdr + _r_val**2 - 6 * _r_val + 1
        )
        _nr_sqrt_s1 = _s_1 * (
            -2 * _nr_sqrt_md2r2 - _nr_sqrt_mdr2 + 2 * _nr_sqrt_mdr + 6 * _r_val - 2
        )
        _nr_sqrt_nos1 = _nr_sqrt_md2r2 - _nr_sqrt_mdr + 1

        _nr_t2_s1 = _nr_sqrt_s1sq + _nr_sqrt_s1 + _nr_sqrt_nos1

        if not np.isclose(
            np.einsum("i->", _nr_t2_mdr.astype(np.float64)),
            np.einsum("i->", _nr_t2_s1.astype(np.float64)),
            rtol=0,
            atol=0.5 * prec,
        ):
            raise RuntimeError(
                "Calculation of sq. root term in exact average GUPPI"
                f"with recapture spec, {f'"{recapture_form}"'} is incorrect."
            )

        _s_2 = (_nr_t1 - np.sqrt(_nr_t2_s1)) / (2 * _r_val)

    else:
        _s_intcpt = mp.fsub(_delta_star + 1 / 2, mp.fabs(_delta_star - 1 / 2))
        _s_2 = (
            (1 / 2)
            + _delta_star
            - _delta_star * _s_1
            - np.sqrt(
                ((_delta_star**2) - 1) * (_s_1**2)
                + (-2 * (_delta_star**2) + _delta_star + 1) * _s_1
                + (_delta_star**2)
                - _delta_star
                + (1 / 4)
            )
        )

    _gbdry_points_inner = np.column_stack((_s_1, _s_2))
    _gbdry_points_end = np.array([(mpf("0.0"), _s_intcpt)], np.float64)

    _gbdry_points = np.vstack((
        _gbdry_points_end,
        np.flip(_gbdry_points_inner, 0),
        _gbdry_points_start,
        np.flip(_gbdry_points_inner, 1),
        np.flip(_gbdry_points_end, 1),
    )).astype(np.float64)
    _s_2 = np.concatenate((np.array([_s_mid], np.float64), _s_2))

    _gbdry_ends = [0, -1]
    _gbdry_odds = np.array(range(1, len(_s_2), 2), np.int64)
    _gbdry_evns = np.array(range(2, len(_s_2), 2), np.int64)

    # Double the are under the curve, and subtract the double counted bit.
    _gbdry_area_simpson = 2 * _gbd_step_sz * (
        (4 / 3) * np.sum(_s_2.take(_gbdry_odds))
        + (2 / 3) * np.sum(_s_2.take(_gbdry_evns))
        + (1 / 3) * np.sum(_s_2.take(_gbdry_ends))
    ) - np.power(_s_mid, 2)

    _s_1_pts, _s_2_pts = np.split(_gbdry_points, 2, axis=1)
    return GuidelinesBoundary(
        np.column_stack((np.array(_s_1_pts), np.array(_s_2_pts))),
        round(float(_gbdry_area_simpson), prec),
    )


def shrratio_boundary_min(
    _delta_star: float = 0.075,
    _r_val: float = 0.85,
    /,
    *,
    recapture_form: str = "inside-out",
    prec: int = 10,
) -> GuidelinesBoundary:
    """
    Share combinations on the minimum GUPPI boundary, with symmetric
    merging-firm margins.

    Notes
    -----
    With symmetric merging-firm margins, the maximum GUPPI boundary is
    defined by the diversion ratio from the smaller merging-firm to the
    larger one, and is hence unaffected by the method of estimating the
    diversion ratio for the larger firm.

    Parameters
    ----------
    _delta_star
        Share ratio (:math:`\\overline{d} / \\overline{r}`).
    _r_val
        Recapture ratio.
    recapture_form
        Whether recapture-ratio is MNL-consistent ("inside-out") or has fixed
        value for both merging firms ("proportional").
    prec
        Number of decimal places for rounding returned shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """

    _delta_star = mpf(f"{_delta_star}")
    _s_intcpt = mpf("1.00")
    _s_mid = _delta_star / (1 + _delta_star)

    if recapture_form == "inside-out":
        # ## Plot envelope of GUPPI boundaries with r_k = r_bar if s_k = min(_s_1, _s_2)
        # ## See (s_i, s_j) in equation~(44), or thereabouts, in paper
        _smin_nr = _delta_star * (1 - _r_val)
        _smax_nr = 1 - _delta_star * _r_val
        _guppi_bdry_env_dr = _smin_nr + _smax_nr
        _s1_pts = np.array(
            (
                0,
                _smin_nr / _guppi_bdry_env_dr,
                _s_mid,
                _smax_nr / _guppi_bdry_env_dr,
                _s_intcpt,
            ),
            np.float64,
        )

        _gbd_area = _s_mid + _s1_pts[1] * (1 - 2 * _s_mid)
    else:
        _s1_pts, _gbd_area = np.array((0, _s_mid, _s_intcpt), np.float64), _s_mid

    return GuidelinesBoundary(
        np.column_stack((_s1_pts, _s1_pts[::-1])), round(float(_gbd_area), prec)
    )


def shrratio_boundary_max(
    _delta_star: float = 0.075, _r_val: float = 0.85, /, *, prec: int = 10
) -> GuidelinesBoundary:
    """
    Share combinations on the minimum GUPPI boundary with symmetric
    merging-firm margins.

    Parameters
    ----------
    _delta_star
        Share ratio (:math:`\\overline{d} / \\overline{r}`).
    _r_val
        Recapture ratio.
    prec
        Number of decimal places for rounding returned shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """

    # _r_val is not needed for max boundary, but is specified for consistency
    # of function call with other shrratio_mgnsym_boundary functions
    del _r_val
    _delta_star = mpf(f"{_delta_star}")
    _s_intcpt = _delta_star
    _s_mid = _delta_star / (1 + _delta_star)

    _s1_pts = (0, _s_mid, _s_intcpt)

    return GuidelinesBoundary(
        np.column_stack((
            np.array(_s1_pts, np.float64),
            np.array(_s1_pts[::-1], np.float64),
        )),
        round(float(_s_intcpt * _s_mid), prec),  # simplified calculation
    )


def _shrratio_boundary_intcpt(
    _s_2_pre: float,
    _delta_star: mpf,
    _r_val: mpf,
    /,
    *,
    recapture_form: Literal["inside-out", "proportional"],
    agg_method: Literal["arithmetic mean", "geometric mean", "distance"],
    weighting: Literal["cross-product-share", "own-share", None],
) -> float:
    match weighting:
        case "cross-product-share":
            _s_intcpt: float = _delta_star
        case "own-share":
            _s_intcpt = mpf("1.0")
        case None if agg_method == "distance":
            _s_intcpt = _delta_star * mp.sqrt("2")
        case None if agg_method == "arithmetic mean" and recapture_form == "inside-out":
            _s_intcpt = mp.fdiv(
                mp.fsub(
                    2 * _delta_star * _r_val + 1, mp.fabs(2 * _delta_star * _r_val - 1)
                ),
                2 * mpf(f"{_r_val}"),
            )
        case None if agg_method == "arithmetic mean" and recapture_form == "proportional":
            _s_intcpt = mp.fsub(_delta_star + 1 / 2, mp.fabs(_delta_star - 1 / 2))
        case _:
            _s_intcpt = _s_2_pre

    return _s_intcpt


def lerp(
    _x1: int | float | mpf | NDArray[np.float64 | np.int64] = 3,
    _x2: int | float | mpf | NDArray[np.float64 | np.int64] = 1,
    _r: float | mpf = 0.25,
    /,
) -> float | mpf | NDArray[np.float64]:
    """
    From the function of the same name in the C++ standard [2]_

    Constructs the weighted average, :math:`w_1 x_1 + w_2 x_2`, where
    :math:`w_1 = 1 - r` and :math:`w_2 = r`.

    Parameters
    ----------
    _x1, _x2
        bounds :math:`x_1, x_2` to interpolate between.
    _r
        interpolation weight :math:`r` assigned to :math:`x_2`

    Returns
    -------
        The linear interpolation, or weighted average,
        :math:`x_1 + r \\cdot (x_1 - x_2) \\equiv (1 - r) \\cdot x_1 + r \\cdot x_2`.

    Raises
    ------
    ValueError
        If the interpolation weight is not in the interval, :math:`[0, 1]`.

    References
    ----------

    .. [2] C++ Reference, https://en.cppreference.com/w/cpp/numeric/lerp

    """

    if not 0 <= _r <= 1:
        raise ValueError("Specified interpolation weight must lie in [0, 1].")
    elif _r == 0:
        return _x1
    elif _r == 1:
        return _x2
    elif _r == 0.5:
        return 1 / 2 * (_x1 + _x2)
    else:
        return _r * _x2 + (1 - _r) * _x1


def round_cust(
    _num: float = 0.060215,
    /,
    *,
    frac: float = 0.005,
    rounding_mode: str = "ROUND_HALF_UP",
) -> float:
    """
    Custom rounding, to the nearest 0.5% by default.

    Parameters
    ----------
    _num
        Number to be rounded.
    frac
        Fraction to be rounded to.
    rounding_mode
        Rounding mode, as defined in the :code:`decimal` package.

    Returns
    -------
        The given number, rounded as specified.

    Raises
    ------
    ValueError
        If rounding mode is not defined in the :code:`decimal` package.

    Notes
    -----
    Integer-round the quotient, :code:`(_num / frac)` using the specified
    rounding mode. Return the product of the rounded quotient times
    the specified precision, :code:`frac`.

    """

    if rounding_mode not in (
        decimal.ROUND_05UP,
        decimal.ROUND_CEILING,
        decimal.ROUND_DOWN,
        decimal.ROUND_FLOOR,
        decimal.ROUND_HALF_DOWN,
        decimal.ROUND_HALF_EVEN,
        decimal.ROUND_HALF_UP,
        decimal.ROUND_UP,
    ):
        raise ValueError(
            f"Value, {f'"{rounding_mode}"'} is invalid for rounding_mode."
            "Documentation for the, \"decimal\" built-in lists valid rounding modes."
        )

    _n, _f, _e = (decimal.Decimal(f"{_g}") for _g in [_num, frac, 1])

    return float(_f * (_n / _f).quantize(_e, rounding=rounding_mode))


def boundary_plot(*, mktshares_plot_flag: bool = True) -> tuple[Any, ...]:
    """Setup basic figure and axes for plots of safe harbor boundaries.

    See, https://matplotlib.org/stable/tutorials/text/pgf.html
    """

    import matplotlib as mpl
    import matplotlib.axes as mpa
    import matplotlib.patches as mpp
    import matplotlib.ticker as mpt

    mpl.use("pgf")
    import matplotlib.pyplot as _plt  # noqa: ICN001

    _plt.rcParams.update({
        "pgf.rcfonts": False,
        "pgf.texsystem": "lualatex",
        "pgf.preamble": "\n".join([
            R"\pdfvariable minorversion=7",
            R"\usepackage{fontspec}",
            R"\usepackage{luacode}",
            R"\begin{luacode}",
            R"local function embedfull(tfmdata)",
            R'    tfmdata.embedding = "full"',
            R"end",
            R"",
            R"luatexbase.add_to_callback("
            R'  "luaotfload.patch_font", embedfull, "embedfull"'
            R")",
            R"\end{luacode}",
            R"\usepackage{mathtools}",
            R"\usepackage{unicode-math}",
            R"\setmathfont[math-style=ISO]{STIX Two Math}",
            R"\setmainfont{STIX Two Text}",
            r"\setsansfont{Fira Sans Light}",
            R"\setmonofont[Scale=MatchLowercase,]{Fira Mono}",
            R"\defaultfontfeatures[\rmfamily]{",
            R"  Ligatures={TeX, Common},",
            R"  Numbers={Proportional, Lining},",
            R"  }",
            R"\defaultfontfeatures[\sffamily]{",
            R"  Ligatures={TeX, Common},",
            R"  Numbers={Monospaced, Lining},",
            R"  LetterSpace=0.50,",
            R"  }",
            R"\usepackage[",
            R"  activate={true, nocompatibility},",
            R"  tracking=true,",
            R"  ]{microtype}",
        ]),
    })

    # Initialize a canvas with a single figure (set of axes)
    _fig = _plt.figure(figsize=(5, 5), dpi=600)
    _ax_out = _fig.add_subplot()

    def _set_axis_def(
        _ax1: mpa.Axes,
        /,
        *,
        mktshares_plot_flag: bool = False,
        mktshares_axlbls_flag: bool = False,
    ) -> mpa.Axes:
        # Set the width of axis gridlines, and tick marks:
        # both axes, both major and minor ticks
        # Frame, grid, and facecolor
        for _spos0 in "left", "bottom":
            _ax1.spines[_spos0].set_linewidth(0.5)
            _ax1.spines[_spos0].set_zorder(5)
        for _spos1 in "top", "right":
            _ax1.spines[_spos1].set_linewidth(0.0)
            _ax1.spines[_spos1].set_zorder(0)
            _ax1.spines[_spos1].set_visible(False)
        _ax1.set_facecolor("#E6E6E6")

        _ax1.grid(linewidth=0.5, linestyle=":", color="grey", zorder=1)
        _ax1.tick_params(axis="x", which="both", width=0.5)
        _ax1.tick_params(axis="y", which="both", width=0.5)

        # Tick marks skip, size, and rotation
        # x-axis
        _plt.setp(
            _ax1.xaxis.get_majorticklabels(),
            horizontalalignment="right",
            fontsize=6,
            rotation=45,
        )
        # y-axis
        _plt.setp(
            _ax1.yaxis.get_majorticklabels(), horizontalalignment="right", fontsize=6
        )

        if mktshares_plot_flag:
            # Axis labels
            if mktshares_axlbls_flag:
                # x-axis
                _ax1.set_xlabel("Firm 1 Market Share, $s_1$", fontsize=10)
                _ax1.xaxis.set_label_coords(0.75, -0.1)
                # y-axis
                _ax1.set_ylabel("Firm 2 Market Share, $s_2$", fontsize=10)
                _ax1.yaxis.set_label_coords(-0.1, 0.75)

            # Plot the ray of symmetry
            _ax1.plot(
                [0, 1], [0, 1], linewidth=0.5, linestyle=":", color="grey", zorder=1
            )

            # Axis scale
            _ax1.set_xlim(0, 1)
            _ax1.set_ylim(0, 1)
            _ax1.set_aspect(1.0)

            # Truncate the axis frame to a triangle:
            _ax1.add_patch(
                mpp.Rectangle(
                    xy=(1.0025, 0.00),
                    width=1.1 * mp.sqrt(2),
                    height=1.1 * mp.sqrt(2),
                    angle=45,
                    color="white",
                    edgecolor=None,
                    fill=True,
                    clip_on=True,
                    zorder=5,
                )
            )
            # Feasible space is bounded by the other diagonal:
            _ax1.plot(
                [0, 1], [1, 0], linestyle="-", linewidth=0.5, color="black", zorder=1
            )

            # Axis Tick-mark locations
            # One can supply an argument to mpt.AutoMinorLocator to
            # specify a fixed number of minor intervals per major interval, e.g.:
            # minorLocator = mpt.AutoMinorLocator(2)
            # would lead to a single minor tick between major ticks.
            _minorLocator = mpt.AutoMinorLocator(5)
            _majorLocator = mpt.MultipleLocator(0.05)
            for _axs in _ax1.xaxis, _ax1.yaxis:
                if _axs == _ax1.xaxis:
                    _majorticklabels_rot = 45
                elif _axs == _ax1.yaxis:
                    _majorticklabels_rot = 0
                # x-axis
                _axs.set_major_locator(_majorLocator)
                _axs.set_minor_locator(_minorLocator)
                # It"s always x when specifying the format
                _axs.set_major_formatter(mpt.StrMethodFormatter("{x:>3.0%}"))

            # Hide every other tick-label
            for _axl in _ax1.get_xticklabels(), _ax1.get_yticklabels():
                _plt.setp(_axl[::2], visible=False)

        return _ax1

    _ax_out = _set_axis_def(_ax_out, mktshares_plot_flag=mktshares_plot_flag)

    return _plt, _fig, _ax_out, _set_axis_def
