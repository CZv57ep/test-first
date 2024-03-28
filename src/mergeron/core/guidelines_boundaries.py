"""
Methods for defining and analyzing boundaries for Guidelines standards,
with a canvas on which to draw boundaries for Guidelines standards.

"""

from dataclasses import dataclass
from importlib.metadata import version
from typing import Literal, TypeAlias

import numpy as np
from attrs import field, frozen
from mpmath import mp, mpf  # type: ignore

from .. import _PKG_NAME, UPPAggrSelector  # noqa: TID252
from . import GuidelinesBoundary, UPPBoundarySpec
from . import guidelines_boundary_functions as gbfn

__version__ = version(_PKG_NAME)


mp.prec = 80
mp.trap_complex = True

HMGPubYear: TypeAlias = Literal[1992, 2004, 2010, 2023]


@dataclass(frozen=True)
class HMGThresholds:
    delta: float
    fc: float
    rec: float
    guppi: float
    divr: float
    cmcr: float
    ipr: float


@frozen
class GuidelinesThresholds:
    """
    Guidelines threholds by Guidelines publication year

    ΔHHI, Recapture Rate, GUPPI, Diversion ratio, CMCR, and IPR thresholds
    constructed from concentration standards in Guidelines published in
    1992, 2004, 2010, and 2023.

    The 2004 Guidelines refernced here are the EU Commission
    guidelines on assessment of horizontal mergers. These
    guidelines also define a presumption for mergers with
    post-merger HHI in [1000, 2000) and ΔHHI >= 250 points,
    whi is not modeled here.

    All other Guidelines modeled here are U.S. merger guidelines.

    """

    pub_year: HMGPubYear
    """
    Year of publication of the Guidelines
    """

    safeharbor: HMGThresholds = field(kw_only=True, default=None)
    """
    Negative presumption quantified on various measures

    ΔHHI safeharbor bound, default recapture rate, GUPPI bound,
    diversion ratio limit, CMCR, and IPR
    """

    imputed_presumption: HMGThresholds = field(kw_only=True, default=None)
    """
    Presumption of harm imputed from guidelines

    ΔHHI bound inferred from strict numbers-equivalent
    of (post-merger) HHI presumption, and corresponding default recapture rate,
    GUPPI bound, diversion ratio limit, CMCR, and IPR
    """

    presumption: HMGThresholds = field(kw_only=True, default=None)
    """
    Presumption of harm defined in HMG

    ΔHHI bound and corresponding default recapture rate, GUPPI bound,
    diversion ratio limit, CMCR, and IPR
    """

    def __attrs_post_init__(self, /) -> None:
        # In the 2023 Guidlines, the agencies do not define a
        # negative presumption, or safeharbor. Practically speaking,
        # given resource constraints and loss aversion, it is likely
        # that staff only investigates mergers that meet the presumption;
        # thus, here, the tentative delta safeharbor under
        # the 2023 Guidelines is 100 points
        _hhi_p, _dh_s, _dh_p = {
            1992: (0.18, 0.005, 0.01),
            2010: (0.25, 0.01, 0.02),
            2004: (0.20, 0.015, 0.015),
            2023: (0.18, 0.01, 0.01),
        }[self.pub_year]

        object.__setattr__(
            self,
            "safeharbor",
            HMGThresholds(
                _dh_s,
                _fc := int(np.ceil(1 / _hhi_p)),
                _r := gbfn.round_cust(_fc / (_fc + 1)),
                _g_s := guppi_from_delta(_dh_s, m_star=1.0, r_bar=_r),
                _dr := gbfn.round_cust(1 / (_fc + 1)),
                _cmcr := 0.03,  # Not strictly a Guidelines standard
                _ipr := _g_s,  # Not strictly a Guidelines standard
            ),
        )

        # imputed_presumption is relevant for 2010 Guidelines
        object.__setattr__(
            self,
            "imputed_presumption",
            (
                HMGThresholds(
                    _dh_i := 2 * (0.5 / _fc) ** 2,
                    _fc,
                    _r_i := gbfn.round_cust((_fc - 1 / 2) / (_fc + 1 / 2)),
                    _g_i := guppi_from_delta(_dh_i, m_star=1.0, r_bar=_r_i),
                    gbfn.round_cust((1 / 2) / (_fc + 1 / 2)),
                    _cmcr,
                    _g_i,
                )
                if self.pub_year == 2010
                else HMGThresholds(
                    _dh_i := 2 * (1 / (_fc + 1)) ** 2,
                    _fc,
                    _r,
                    _g_i := guppi_from_delta(_dh_i, m_star=1.0, r_bar=_r),
                    _dr,
                    _cmcr,
                    _g_i,
                )
            ),
        )

        object.__setattr__(
            self,
            "presumption",
            HMGThresholds(
                _dh_p,
                _fc,
                _r,
                _g_p := guppi_from_delta(_dh_p, m_star=1.0, r_bar=_r),
                _dr,
                _cmcr,
                _ipr := _g_p,
            ),
        )


def guppi_from_delta(
    _delta_bound: float = 0.01, /, *, m_star: float = 1.00, r_bar: float = 0.855
) -> float:
    """
    Translate ∆HHI bound to GUPPI bound.

    Parameters
    ----------
    _deltasf
        Specified ∆HHI bound.
    m_star
        Parametric price-cost margin.
    r_bar
        Default recapture rate.

    Returns
    -------
        GUPPI bound corresponding to ∆HHI bound, at given margin and recapture rate.

    """
    return gbfn.round_cust(
        m_star * r_bar * (_s_m := np.sqrt(_delta_bound / 2)) / (1 - _s_m),
        frac=0.005,
        rounding_mode="ROUND_HALF_DOWN",
    )


def critical_share_ratio(
    _guppi_bound: float = 0.075,
    /,
    *,
    m_star: float = 1.00,
    r_bar: float = 1.00,
    frac: float = 1e-16,
) -> mpf:
    """
    Corollary to GUPPI bound.

    Parameters
    ----------
    _guppi_bound
        Specified GUPPI bound.
    m_star
        Parametric price-cost margin.
    r_bar
        Default recapture rate.

    Returns
    -------
        Critical share ratio (share ratio bound) corresponding to the GUPPI bound
        for given margin and recapture rate.

    """
    return gbfn.round_cust(
        mpf(f"{_guppi_bound}") / mp.fmul(f"{m_star}", f"{r_bar}"), frac=frac
    )


def share_from_guppi(
    _guppi_bound: float = 0.065, /, *, m_star: float = 1.00, r_bar: float = 0.855
) -> float:
    """
    Symmetric-firm share for given GUPPI, margin, and recapture rate.

    Parameters
    ----------
    _guppi_bound
        GUPPI bound.
    m_star
        Parametric price-cost margin.
    r_bar
        Default recapture rate.

    Returns
    -------
    float
        Symmetric firm market share on GUPPI boundary, for given margin and
        recapture rate.

    """

    return gbfn.round_cust(
        (_d0 := critical_share_ratio(_guppi_bound, m_star=m_star, r_bar=r_bar))
        / (1 + _d0)
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
    _dh_half = np.row_stack((
        np.column_stack((_s_1, _s_2)),
        np.array([(mpf("0.0"), mpf("1.0"))]),
    ))
    _dh_bdry_pts = np.row_stack((np.flip(_dh_half, 0), np.flip(_dh_half[1:], 1)))

    _s_1_pts, _s_2_pts = np.split(_dh_bdry_pts, 2, axis=1)
    return GuidelinesBoundary(
        np.column_stack((
            np.array(_s_1_pts, np.float64),
            np.array(_s_2_pts, np.float64),
        )),
        gbfn.dh_area(_dh_val, prec=prec),
    )


def combined_share_boundary(
    _s_intcpt: float = 0.0625, /, *, bdry_dps: int = 10
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
    bdry_dps
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
        round(float(_s_intcpt * _s_mid), bdry_dps),
    )


def hhi_pre_contrib_boundary(
    _hhi_contrib: float = 0.03125, /, *, bdry_dps: int = 5
) -> GuidelinesBoundary:
    """
    Share combinations on the premerger HHI contribution boundary.

    Parameters
    ----------
    _hhi_contrib:
        Merging-firms' pre-merger HHI contribution bound.
    bdry_dps
        Number of decimal places for rounding reported shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """
    _hhi_contrib = mpf(f"{_hhi_contrib}")
    _s_mid = mp.sqrt(_hhi_contrib / 2)

    _bdry_step_sz = mp.power(10, -bdry_dps)
    # Range-limit is 0 less a step, which is -1 * step-size
    _s_1 = np.array(mp.arange(_s_mid, -_bdry_step_sz, -_bdry_step_sz), np.float64)
    _s_2 = np.sqrt(_hhi_contrib - _s_1**2).astype(np.float64)
    _bdry_pts_mid = np.column_stack((_s_1, _s_2))
    return GuidelinesBoundary(
        np.row_stack((np.flip(_bdry_pts_mid, 0), np.flip(_bdry_pts_mid[1:], 1))),
        round(float(mp.pi * _hhi_contrib / 4), bdry_dps),
    )


def hhi_post_contrib_boundary(
    _hhi_contrib: float = 0.800, /, *, bdry_dps: int = 10
) -> GuidelinesBoundary:
    """
    Share combinations on the postmerger HHI contribution boundary.

    The post-merger HHI contribution boundary is identical to the
    combined-share boundary.

    Parameters
    ----------
    _hhi_contrib:
        Merging-firms' pre-merger HHI contribution bound.
    bdry_dps
        Number of decimal places for rounding reported shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """
    return combined_share_boundary(np.sqrt(_hhi_contrib), bdry_dps=bdry_dps)


def diversion_ratio_boundary(_bdry_spec: UPPBoundarySpec) -> GuidelinesBoundary:
    _share_ratio = critical_share_ratio(
        _bdry_spec.diversion_ratio, r_bar=_bdry_spec.rec
    )
    match _bdry_spec.agg_method:
        case UPPAggrSelector.AVG:
            return gbfn.shrratio_boundary_xact_avg(
                _share_ratio,
                _bdry_spec.rec,
                recapture_form=_bdry_spec.recapture_form.value,  # type: ignore
                prec=_bdry_spec.precision,
            )
        case UPPAggrSelector.MAX:
            return gbfn.shrratio_boundary_max(
                _share_ratio, _bdry_spec.rec, prec=_bdry_spec.precision
            )
        case UPPAggrSelector.MIN:
            return gbfn.shrratio_boundary_min(
                _share_ratio,
                _bdry_spec.rec,
                recapture_form=_bdry_spec.recapture_form.value,  # type: ignore
                prec=_bdry_spec.precision,
            )
        case UPPAggrSelector.DIS:
            return gbfn.shrratio_boundary_wtd_avg(
                _share_ratio,
                _bdry_spec.rec,
                agg_method="distance",
                weighting=None,
                recapture_form=_bdry_spec.recapture_form.value,  # type: ignore
                prec=_bdry_spec.precision,
            )
        case _:
            _weighting = (
                "cross-product-share"
                if _bdry_spec.agg_method.value.startswith("cross-product-share")
                else "own-share"
            )

            _agg_method = (
                "arithmetic"
                if _bdry_spec.agg_method.value.endswith("average")
                else "distance"
            )

            return gbfn.shrratio_boundary_wtd_avg(
                _share_ratio,
                _bdry_spec.rec,
                agg_method=_agg_method,  # type: ignore
                weighting=_weighting,  # type: ignore
                recapture_form=_bdry_spec.recapture_form.value,  # type: ignore
                prec=_bdry_spec.precision,
            )
