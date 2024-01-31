"""
Functions for defining and analyzing boundaries for Guidelines standards,
with a canvas on which to draw boundaries for Guidelines standards.

"""

import decimal
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from mpmath import mp, mpf  # type: ignore
from numpy.typing import NDArray

__version__ = metadata.version(Path(__file__).parents[1].stem)

mp.prec = 80
mp.trap_complex = True

recapture_spec_values = ("inside-out", "proportional")

# In the 2023 Guidlines, the agencies do not specify a safeharbor
#   Practically speaking, given resource constraints and loss aversion,
#   it is likely that staff only investigates mergers that meet the presumption
#   as evidenced by the data for 2003--2011
guidelines_bounds_dict = MappingProxyType({
    1992: (0.18, 0.005, 0.01),
    2010: (0.25, 0.01, 0.02),
    2023: (0.18, 0.01, 0.01),
})  #: maps _pub_year to (post-merger HHI threshold, delta HHI bound)


@dataclass
class GuidelinesStandards:
    """
    Defines Guidelines standards by Guidelines publication year based on
    concentration standards in U.S. Horizontal Merger Guidelines.

    Attributes
    ----------
    pub_year: Literal[1992, 2010, 2023]
        Year of publication of the U.S. Horizontal Merger Guidelines (HMG);
        1992, 2010, or 2023
    safeharbor: tuple
        ΔHHI safeharbor bound, default recapture rate, GUPPI bound and
        diversion ratio limit at ΔHHI safeharbor
    strict_presumption: tuple
        ΔHHI safeharbor bound, default recapture rate, GUPPI bound and
        diversion ratio limit at enforcement margin for Guidelines
        presumption of harm, interpreted strictly
    presumption: tuple
        ΔHHI safeharbor bound, default recapture rate, GUPPI bound and
        diversion ratio limit at enforcement margin for Guidelines
        presumption of harm, as typically interpreted in the literature
        on merger enforcement
    """

    def __init__(self, _pub_year: Literal[1992, 2010, 2023] = 1992, /):
        """

        Parameters
        ----------
        _pub_year
            Guidelines publication year, 1992 or 2010

        Raises
        ------
        ValueError
            If pub_year is invalid

        """
        if _pub_year not in (1992, 2010, 2023):
            raise ValueError("Argument must be 1992, 2010 or 2023")
        self.pub_year: Literal[1992, 2010, 2023] = _pub_year

        _hhi, _dh_s, _dh_p = guidelines_bounds_dict[_pub_year]
        self.safeharbor: Final = (
            _dh_s,
            _fc := np.ceil(1 / _hhi),
            _r := round_cust(_fc / (_fc + 1)),
            _g_s := gbd_from_dsf(_dh_s, m_star=1.0, r_bar=_r),
            _dr := round_cust(1 / (_fc + 1)),
            _cmcr := 0.03,  #: Not strictly a Guidelines standard
            _ipr := _g_s,  #: Not strictly a Guidelines standard
        )
        self.inferred_presumption: Final = (
            (
                _dh := 2 * (_s := 1 / (_fc + 1)) * _s,
                _fc,
                _r,
                _g_p := gbd_from_dsf(_dh, m_star=1.0, r_bar=_r),
                _dr,
                _cmcr,
                _ipr := _g_p,
            )
            if _pub_year in (1992, 2023)
            else (
                _dh := 2 * (_s := 0.5 / _fc) * _s,
                _fc,
                _r,  # _rs := round_cust((_fc - 1 / 2) / (_fc + 1 / 2)),
                _g_p := gbd_from_dsf(_dh, m_star=1.0, r_bar=_r),
                round_cust((1 / 2) / (_fc - 1 / 2)),
                _cmcr,
                _ipr := _g_p,
            )
        )
        self.presumption: Final = (_dh_p, _fc, _r, _g_p, _dr, _cmcr, _ipr)


def round_cust(
    _num: float = 0.060215, /, *, frac: float = 0.005, mode: str = "ROUND_HALF_UP"
) -> float:
    """
    Custom rounding, to the nearest 0.5% by default.

    Parameters
    ----------
    _num
        Number to be rounded.
    frac
        Fraction to be rounded to.
    mode
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
    Round a decimal fraction down or up to specified precision, based on
    halving the remainder after division of the given decimal fraction
    by the specified precision expressed as another decimal fraction.

    """

    if mode not in (
        decimal.ROUND_HALF_UP,
        decimal.ROUND_DOWN,
        decimal.ROUND_05UP,
        decimal.ROUND_CEILING,
        decimal.ROUND_FLOOR,
        decimal.ROUND_HALF_DOWN,
        decimal.ROUND_HALF_EVEN,
        decimal.ROUND_UP,
    ):
        raise ValueError(f"Value, {mode!r} of mode is invalid.")

    _num, _frac = (decimal.Decimal(f"{_g}") for _g in (_num, frac))

    return float(_frac * (_num / _frac).quantize(0, rounding=mode))


def lerp(
    _x1: int | float | mpf | NDArray[np.float_ | np.int_] = 3,
    _x2: int | float | mpf | NDArray[np.float_ | np.int_] = 1,
    _r: float | mpf = 0.25,
    /,
) -> float | mpf | NDArray[np.float_]:
    """
    From the C++ standard function of the same name.

    Constructs the weighted average, :math:`w_1 x_1 + w_2 x_2`, where
    :math:`w_1 = 1 - r` and :math:`w_2 = r`.

    Parameters
    ----------
    _x1, _x2
        bounds :math:`x_1, x_2` to interpolate between.
    _r
        interpolation weight :math:`r` assigned to :math:`x_2`.

    Returns
    -------
        The linear interpolation, or weighted average, :math:`x_1 (1 - r) + x_2 r`.

    Raises
    ------
    ValueError
        If the interpolation weight is not in the interval, :math:`[0, 1]`.

    """
    if not 0 <= _r <= 1:
        raise ValueError("Specified interpolation weight must lie in [0, 1].")
    return _x2 * _r + _x1 * (1 - _r)


def gbd_from_dsf(
    _deltasf: float = 0.01, /, *, m_star: float = 1.00, r_bar: float = 0.80
) -> float:
    """
    Translate ∆HHI bound to GUPPI bound.

    Parameters
    ----------
    _deltasf
        Specified ∆HHI bound.
    m_star
        Margin level.
    r_bar
        Default (fixed) recapture rate.

    Returns
    -------
        GUPPI bound corresponding to ∆HHI bound, at given margin and recapture rate.

    """
    return round_cust(
        m_star * r_bar * (_s_m := np.sqrt(_deltasf / 2)) / (1 - _s_m),
        frac=0.005,
        mode="ROUND_HALF_DOWN",
    )


def critical_shrratio(
    _gbd: float = 0.06, _m_val: float = 1.00, _r_val: float = 0.80, /
) -> mpf:
    """
    Corollary to GUPPI bound.

    Parameters
    ----------
    _gbd
        Specified GUPPI bound.
    _m_val
        Margin level.
    _r_val
        Default (fixed) recapture rate.

    Returns
    -------
        Critical share ratio (share ratio bound) corresponding to the GUPPI bound
        for given margin and recapture rate.

    """
    return mpf(f"{_gbd}") / mp.fmul(f"{_m_val}", f"{_r_val}")


def shr_from_gbd(
    _gbd: float = 0.06, /, *, m_star: float = 1.00, r_bar: float = 0.80
) -> float:
    """
    Symmetric-firm share for given GUPPI, margin, and recapture rate.

    Parameters
    ----------
    _gbd
        GUPPI bound.
    m_star
        Margin level.
    r_bar
        Default (fixed) recapture rate.

    Returns
    -------
        Symmetric firm market share on GUPPI boundary, for given margin and
        recapture rate.

    """

    return round_cust((_d0 := critical_shrratio(_gbd, m_star, r_bar)) / (1 + _d0))


def boundary_plot(*, share_axes_flag: bool = True) -> tuple[Any, ...]:
    """Setup basic figure and axes for plots of safe harbor boundaries.

    See, https://matplotlib.org/stable/tutorials/text/pgf.html
    """

    # from matplotlib.backends.backend_pgf import FigureCanvasPgf
    mpl.backend_bases.register_backend("pdf", FigureCanvasPgf)
    # mpl.use("pgf")
    # import matplotlib.pyplot as plt

    plt.rcParams.update({
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
            R"	Ligatures={TeX, Common},",
            R"	Numbers={Proportional, Lining},",
            R"	}",
            R"\defaultfontfeatures[\sffamily]{",
            R"	Ligatures={TeX, Common},",
            R"	Numbers={Monospaced, Lining},",
            R"	LetterSpace=0.50,",
            R"	}",
            R"\usepackage[",
            R"	activate={true, nocompatibility},",
            R"	tracking=true,",
            R"	]{microtype}",
        ]),
    })

    # Initialize a canvas with a single figure (set of axes)
    _fig = plt.figure(figsize=(5, 5), dpi=600)
    _ax_out = _fig.add_subplot()

    def _set_axis_def(
        _ax1: mpl.axes._axes.Axes, /, share_axes_flag: bool = False
    ) -> mpl.axes.Axes:
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
        del _spos0, _spos1
        _ax1.set_facecolor("#F6F6F6")

        _ax1.grid(linewidth=0.5, linestyle=":", color="grey", zorder=1)
        _ax1.tick_params(axis="x", which="both", width=0.5)
        _ax1.tick_params(axis="y", which="both", width=0.5)

        # Tick marks skip, size, and rotation
        # x-axis
        plt.setp(
            _ax1.xaxis.get_majorticklabels(),
            horizontalalignment="right",
            fontsize=6,
            rotation=45,
        )
        # y-axis
        plt.setp(
            _ax1.yaxis.get_majorticklabels(), horizontalalignment="right", fontsize=6
        )

        # Axis labels
        # x-axis
        _ax1.set_xlabel("Firm 1 Market Share, $s_1$", fontsize=10)
        _ax1.xaxis.set_label_coords(0.75, -0.1)
        # y-axis
        _ax1.set_ylabel("Firm 2 Market Share, $s_2$", fontsize=10)
        _ax1.yaxis.set_label_coords(-0.1, 0.75)

        if share_axes_flag:
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
                mpl.patches.Rectangle(
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
            # One can supply an argument to mpl.ticker.AutoMinorLocator to
            # specify a fixed number of minor intervals per major interval, e.g.:
            # minorLocator = mpl.ticker.AutoMinorLocator(2)
            # would lead to a single minor tick between major ticks.
            _minorLocator = mpl.ticker.AutoMinorLocator(5)
            _majorLocator = mpl.ticker.MultipleLocator(0.05)
            for _axs in _ax1.xaxis, _ax1.yaxis:
                if _axs == _ax1.xaxis:
                    _majorticklabels_rot = 45
                elif _axs == _ax1.yaxis:
                    _majorticklabels_rot = 0
                # x-axis
                _axs.set_major_locator(_majorLocator)
                _axs.set_minor_locator(_minorLocator)
                # It"s always x when specifying the format
                _axs.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:>3.0%}"))

            # Hide every other tick-label
            for _axl in _ax1.get_xticklabels(), _ax1.get_yticklabels():
                plt.setp(_axl[::2], visible=False)

        return _ax1

    _ax_out = _set_axis_def(_ax_out, share_axes_flag=share_axes_flag)

    return plt, _fig, _ax_out, _set_axis_def


def dh_area(_dh_val: float = 0.01, /, *, dh_dps: int = 9) -> float:
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
    dh_dps
        Specified precision in decimal places.

    Returns
    -------
        Area under ΔHHI boundary.

    """

    _dh_val = mpf(f"{_dh_val}")
    _s1_zero = (1 - mp.sqrt(1 - 2 * _dh_val)) / 2
    _s1_one = 1 - _s1_zero

    return round(
        float(_s1_zero + (_dh_val / 2) * (mp.ln(_s1_one) - mp.ln(_s1_zero))), dh_dps
    )


def dh_area_quad(_dh_val: float = 0.01, /, *, dh_dps: int = 9) -> float:
    """
    Area under the ΔHHI boundary.

    When the given ΔHHI bound matches a Guidelines safeharbor,
    the area under the boundary is half the intrinsic clearance rate
    for the ΔHHI safeharbor.

    Parameters
    ----------
    _dh_val
        Merging-firms' ΔHHI bound.
    dh_dps
        Specified precision in decimal places.

    Returns
    -------
        Area under ΔHHI boundary.

    """

    _dh_val = mpf(f"{_dh_val}")
    _s1_zero = (1 - mp.sqrt(1 - 2 * _dh_val)) / 2
    _s1_one = 1 - _s1_zero

    return round(
        float(_s1_zero + mp.quad(lambda x: _dh_val / (2 * x), [_s1_zero, _s1_one])),
        dh_dps,
    )


def delta_hhi_boundary(
    _dh_val: float = 0.01, /, *, dh_dps: int = 5
) -> tuple[NDArray[np.float_], float]:
    """
    Generate the list of share combination on the ΔHHI boundary.

    Parameters
    ----------
    _dh_val:
        Merging-firms' ΔHHI bound.
    dh_dps
        Number of decimal places for rounding reported shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """

    _dh_val = mpf(f"{_dh_val}")
    _s1_zero = 1 / 2 * (1 - mp.sqrt(1 - 2 * _dh_val))
    _s1_one = 1 - _s1_zero

    _s_mid = mp.sqrt(_dh_val / 2)

    _dh_step_sz = mp.power(10, -6)
    _s_1 = np.array(mp.arange(_s_mid, _s1_zero, -_dh_step_sz))
    _s_2 = _dh_val / (2 * _s_1)

    # Boundary points
    _dh_half = np.row_stack((
        np.column_stack((_s_1, _s_2)),
        np.array([(_s1_zero, _s1_one)]),
        np.array([(mpf("0.0"), mpf("1.0"))]),
    ))
    _dh_bdry_pts = np.row_stack((np.flip(_dh_half, 0), np.flip(_dh_half[1:], 1)))

    _s_1_pts, _s_2_pts = np.split(_dh_bdry_pts, 2, axis=1)
    return (
        np.column_stack((np.array(_s_1_pts, np.float_), np.array(_s_2_pts, np.float_))),
        dh_area(_dh_val, dh_dps=dh_dps),
    )


def combined_share_boundary(
    _s_incpt: float = 0.0625, /, *, bdry_dps: int = 10
) -> tuple[NDArray[np.float_], float]:
    """
    Share combinations on the merging-firms' combined share boundary.

    Assumes symmetric merging-firm margins. The combined-share is
    congruent to the post-merger HHI contribution boundary, as the
    post-merger HHI bound is the square of the combined-share bound.

    Parameters
    ----------
    _s_incpt:
        Merging-firms' combined share.
    bdry_dps
        Number of decimal places for rounding reported shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """
    _s_incpt = mpf(f"{_s_incpt}")
    _s_mid = _s_incpt / 2

    _s1_pts = (0, _s_mid, _s_incpt)
    return (
        np.column_stack((
            np.array(_s1_pts, np.float_),
            np.array(_s1_pts[::-1], np.float_),
        )),
        round(float(_s_incpt * _s_mid), bdry_dps),
    )


def hhi_pre_contrib_boundary(
    _hhi_contrib: float = 0.03125, /, *, bdry_dps: int = 5
) -> tuple[NDArray[np.float_], float]:
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
    _s_1 = np.array(mp.arange(_s_mid, -_bdry_step_sz, -_bdry_step_sz), np.float_)
    _s_2 = np.sqrt(_hhi_contrib - _s_1**2).astype(np.float_)
    _bdry_pts_mid = np.column_stack((_s_1, _s_2))
    return (
        np.row_stack((np.flip(_bdry_pts_mid, 0), np.flip(_bdry_pts_mid[1:], 1))),
        round(float(mp.pi * _hhi_contrib / 4), bdry_dps),
    )


def shrratio_mgnsym_boundary_max(
    _delta_star: float = 0.075, _r_val: float = 0.80, /, *, gbd_dps: int = 10
) -> tuple[NDArray[np.float_], float]:
    """
    Share combinations on the minimum GUPPI boundary with symmetric
    merging-firm margins.

    Parameters
    ----------
    _delta_star
        Margin-adjusted benchmark share ratio.
    _r_val
        Recapture ratio.
    gbd_dps
        Number of decimal places for rounding returned shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """

    if _delta_star > 1:
        raise ValueError(
            "Invalid combination specified; "
            "Margin-adjusted benchmark share ratio cannot exceed 1."
        )

    # _r_val is not needed for max boundary, but is specified for consistency
    # of function call with other shrratio_mgnsym_boundary functions
    del _r_val
    _delta_star = mpf(f"{_delta_star}")
    _s_incpt = _delta_star
    _s_mid = _delta_star / (1 + _delta_star)

    _s1_pts = (0, _s_mid, _s_incpt)

    return (
        np.column_stack((
            np.array(_s1_pts, np.float_),
            np.array(_s1_pts[::-1], np.float_),
        )),
        round(float(_s_incpt * _s_mid), gbd_dps),  # simplified calculation
    )


def shrratio_mgnsym_boundary_min(
    _delta_star: float = 0.075,
    _r_val: float = 0.80,
    /,
    *,
    recapture_spec: str = "inside-out",
    gbd_dps: int = 10,
) -> tuple[NDArray[np.float_], float]:
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
        Margin-adjusted benchmark share ratio.
    _r_val
        Recapture ratio.
    recapture_spec
        Whether recapture-ratio is MNL-consistent ("inside-out") or has fixed
        value for both merging firms ("proportional").
    gbd_dps
        Number of decimal places for rounding returned shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """

    if _delta_star > 1:
        raise ValueError("Margin-adjusted benchmark share ratio cannot exceed 1.")

    if recapture_spec not in (_recspecs := ("inside-out", "proportional")):
        raise ValueError(
            f"Recapture_spec value, {recapture_spec!r} not in {_recspecs!r}"
        )

    _delta_star = mpf(f"{_delta_star}")
    _s_incpt = mpf("1.00")
    _s_mid = _delta_star / (1 + _delta_star)

    if recapture_spec == "inside-out":
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
                _s_incpt,
            ),
            np.float_,
        )

        _gbd_area = (_smin_nr + (_smax_nr - _smin_nr) * _s_mid) / _guppi_bdry_env_dr
    else:
        _s1_pts, _gbd_area = np.array((0, _s_mid, _s_incpt), np.float_), _s_mid

    return np.column_stack((_s1_pts, _s1_pts[::-1])), round(float(_gbd_area), gbd_dps)


def shrratio_mgnsym_boundary_wtd_avg(
    _delta_star: float = 0.075,
    _r_val: float = 0.80,
    /,
    *,
    avg_method: Literal["arithmetic", "geometric", "distance"] = "arithmetic",
    wgtng_policy: Literal["own-share", "cross-product-share"] = "own-share",
    recapture_spec: Literal["inside-out", "proportional"] = "inside-out",
    gbd_dps: int = 5,
) -> tuple[NDArray[np.float_], float]:
    """
    Share combinations for the share-weighted average GUPPI boundary with symmetric
    merging-firm margins.

    Notes
    -----
    An analytical expression for the share-weighted arithmetic mean boundary
    is derived and plotted from y-intercept to the ray of symmetry as follows::

        from sympy import plot as symplot, solve, symbols
        s_1, s_2, delta_star = symbols("s_1 s_2 \\delta^*")

        g_val, r_val, m_val = 0.06, 0.80, 0.30
        d_hat = g_val / (r_val * m_val)

        # recapture_spec == "inside-out"
        oswag = solve(
            s_1 * s_2 / (1 - s_1)
            + s_2 * s_1 / (1 - (r_val * s_2 + (1 - r_val) * s_1))
            - (s_1 + s_2) * delta_star,
            s_2
        )[0]
        symplot(
            oswag.subs({delta_star: d_hat}),
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
            cpwag.subs({delta_star: d_hat}),
            (s_1, 0., d_hat / (1 + d_hat)),
            ylabel=s_2
        )

        # recapture_spec == "proportional"
        oswag = solve(
            s_1 * s_2 / (1 - s_1)
            + s_2 * s_1 / (1 - s_2)
            - (s_1 + s_2) * delta_star,
             s_2
        )[0]
        symplot(
            oswag.subs({delta_star: d_hat}),
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
            cpswag.subs({delta_star: d_hat}),
            (s_1, 0.0, d_hat / (1 + d_hat)),
            ylabel=s_2
        )

    Parameters
    ----------
    _delta_star
        GUPPI bound
    _r_val
        Recapture ratio.
    gbd_dps
        Number of decimal places for rounding returned shares.
    avg_method
        Whether "arithmetic", "geometric", or "distance".
    wgtng_policy
        Whether "own-share" or "cross-product-share".
    recapture_spec
        Whether recapture-ratio is MNL-consistent ("inside-out") or has fixed
        value for both merging firms ("proportional").
    gbd_dps
        Number of decimal places for rounding returned shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """

    if _delta_star > 1:
        raise ValueError(
            "Invalid combination specified; "
            "Margin-adjusted benchmark share ratio cannot exceed 1."
        )

    _delta_star = mpf(f"{_delta_star}")
    _s_mid = _delta_star / (1 + _delta_star)

    _gbdry_points = [(_s_mid, _s_mid)]
    _s_1_pre, _s_2_pre = _s_mid, _s_mid
    _s_2_oddval, _s_2_oddsum, _s_2_evnsum = True, 0, 0

    _gbd_step_sz = mp.power(10, -gbd_dps)
    _theta = _gbd_step_sz * (10 if wgtng_policy == "cross-product-share" else 1)
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
                if recapture_spec == "inside-out"
                else _s_1 / (1 - _s_2)
            )

            _r = mp.fdiv(
                _s_1 if wgtng_policy == "cross-product-share" else _s_2, _s_1 + _s_2
            )

            match avg_method:
                case "geometric":
                    _delta_test = mp.expm1(lerp(mp.log1p(_de_1), mp.log1p(_de_2), _r))
                case "distance":
                    _delta_test = mp.sqrt(lerp(_de_1**2, _de_2**2, _r))
                case _:
                    _delta_test = lerp(_de_1, _de_2, _r)

            if wgtng_policy == "cross-product-share":
                test_flag, incr_decr = (_delta_test > _delta_star, -1)
            else:
                test_flag, incr_decr = (_delta_test < _delta_star, 1)

            if test_flag:
                _s_2 += incr_decr * _gbd_step_sz
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

    _gbd_prtlarea = _gbd_step_sz * (
        (4 * _s_2_oddsum + 2 * _s_2_evnsum + _s_mid + _delta_star) / 3
        if wgtng_policy == "cross-product-share"
        else (
            (4 * _s_2_oddsum + 2 * _s_2_evnsum + _s_mid + _s_2_pre) / 3
            + _s_1_pre * (1 + _s_2_pre) / 2
        )
    )

    # Area under boundary
    _gbdry_area_total = 2 * _gbd_prtlarea - mp.power(_s_mid, 2)

    _gbdry_points = np.row_stack((
        _gbdry_points,
        (
            mpf("0.0"),
            _delta_star if wgtng_policy == "cross-product-share" else mpf("1.0"),
        ),
    )).astype(np.float_)
    # Points defining boundary to point-of-symmetry
    return (
        np.row_stack((np.flip(_gbdry_points, 0), np.flip(_gbdry_points[1:], 1))),
        round(float(_gbdry_area_total), gbd_dps),
    )


def shrratio_mgnsym_boundary_xact_avg(
    _delta_star: float = 0.075,
    _r_val: float = 0.80,
    /,
    *,
    recapture_spec: Literal["inside-out", "proportional"] = "inside-out",
    gbd_dps: int = 5,
) -> tuple[NDArray[np.float_], float]:
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

        # recapture_spec = "inside-out"
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

        # recapture_spec = "proportional"
        sag = solve((s_2/(1 - s_1)) + (s_1/(1 - s_2)) - 2 * d_hat, s_2)[0]
        symplot(
            sag,
            (s_1, 0., d_hat / (1 + d_hat)),
            ylabel=s_2
        )

    Parameters
    ----------
    _delta_star
        Margin-adjusted benchmark share ratio.
    _r_val
        Recapture ratio
    recapture_spec
        Whether recapture-ratio is MNL-consistent ("inside-out") or has fixed
        value for both merging firms ("proportional").
    gbd_dps
        Number of decimal places for rounding returned shares.

    Returns
    -------
        Array of share-pairs, area under boundary, area under boundary.

    """

    if _delta_star > 1:
        raise ValueError(
            "Invalid combination specified; "
            "Margin-adjusted benchmark share ratio cannot exceed 1."
        )

    if recapture_spec not in (_recspecs := ("inside-out", "proportional")):
        raise ValueError(f"Recapture spec must be one of {_recspecs:!}")

    _delta_star = mpf(f"{_delta_star}")
    _s_mid = _delta_star / (1 + _delta_star)
    _gbd_step_sz = mp.power(10, -gbd_dps)

    _gbdry_points_start = np.array([(_s_mid, _s_mid)])
    _s_1 = np.array(mp.arange(_s_mid - _gbd_step_sz, 0, -_gbd_step_sz), np.float_)
    if recapture_spec == "inside-out":
        _s_incpt = mp.fdiv(
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
            np.einsum("i->", _nr_t2_mdr.astype(np.float_)),
            np.einsum("i->", _nr_t2_s1.astype(np.float_)),
            rtol=0,
            atol=2 * gbd_dps,
        ):
            raise RuntimeError(
                "Calculation of sq. root term in exact average GUPPI"
                f"with recapture spec, {recapture_spec!r} is incorrect."
            )

        _s_2 = (_nr_t1 - np.sqrt(_nr_t2_s1)) / (2 * _r_val)

    else:
        _s_incpt = mp.fsub(_delta_star + 1 / 2, mp.fabs(_delta_star - 1 / 2))
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
    _gbdry_points_end = np.array([(mpf("0.0"), _s_incpt)], np.float_)

    _gbdry_points = np.row_stack((
        _gbdry_points_end,
        np.flip(_gbdry_points_inner, 0),
        _gbdry_points_start,
        np.flip(_gbdry_points_inner, 1),
        np.flip(_gbdry_points_end, 1),
    )).astype(np.float_)
    _s_2 = np.concatenate((np.array([_s_mid], np.float_), _s_2))

    _gbdry_ends = [0, -1]
    _gbdry_odds = np.array(range(1, len(_s_2), 2), np.int_)
    _gbdry_evns = np.array(range(2, len(_s_2), 2), np.int_)

    # Double the are under the curve, and subtract the double counted bit.
    _gbdry_area_simpson = 2 * _gbd_step_sz * (
        (4 / 3) * np.sum(_s_2.take(_gbdry_odds))
        + (2 / 3) * np.sum(_s_2.take(_gbdry_evns))
        + (1 / 3) * np.sum(_s_2.take(_gbdry_ends))
    ) - np.power(_s_mid, 2)

    _s_1_pts, _s_2_pts = np.split(_gbdry_points, 2, axis=1)
    return (
        np.column_stack((np.array(_s_1_pts), np.array(_s_2_pts))),
        round(float(_gbdry_area_simpson), gbd_dps),
    )


def shrratio_mgnsym_boundary_avg(
    _delta_star: float = 0.075,
    _r_val: float = 0.80,
    /,
    *,
    avg_method: Literal["arithmetic", "geometric", "distance"] = "arithmetic",
    recapture_spec: Literal["inside-out", "proportional"] = "inside-out",
    gbd_dps: int = 5,
) -> tuple[NDArray[np.float_], float]:
    """
    Share combinations along the average GUPPI boundary, with
    symmetric merging-firm margins.

    Parameters
    ----------
    _delta_star
        Margin-adjusted benchmark share ratio.
    _r_val
        Recapture ratio.
    avg_method
        Whether "arithmetic", "geometric", or "root-mean-square".
    recapture_spec
        Whether recapture-ratio is MNL-consistent ("inside-out") or has fixed
        value for both merging firms ("proportional").
    gbd_dps
        Number of decimal places for rounding returned shares.

    Returns
    -------
        Array of share-pairs, area under boundary.

    """

    if _delta_star > 1:
        raise ValueError(
            "Invalid combination specified; "
            "Margin-adjusted benchmark share ratio cannot exceed 1."
        )

    if avg_method not in (_avgmthds := ("arithmetic", "geometric", "root-mean-square")):
        raise ValueError(
            "{} {}".format(
                f"Averarging method, {avg_method!r} is invalid.",
                f"Must be one of {_avgmthds!r}.",
            )
        )

    if recapture_spec not in (_recspecs := ("inside-out", "proportional")):
        raise ValueError(
            "{} {}".format(
                f"Recapture spec, {recapture_spec!r} is invalid.",
                f"Must be one of {_recspecs!r}.",
            )
        )

    _delta_star = mpf(f"{_delta_star}")
    _s_mid = _delta_star / (1 + _delta_star)
    _gbd_step_sz = mp.power(10, -gbd_dps)

    _s_2 = _s_mid
    _s_2_oddval = True
    _s_2_oddsum = 0
    _s_2_evnsum = 0
    _gbdry_points = [(_s_mid, _s_mid)]
    for _s_1 in mp.arange(_s_mid, 0, -_gbd_step_sz):
        _s_1 -= _gbd_step_sz
        while True:
            _shratio_12 = _s_2 / (1 - _s_1)
            _shratio_21 = (
                _s_1 / (1 - _s_2)
                if recapture_spec == "proportional"
                else _s_1 / (1 - lerp(_s_1, _s_2, _r_val))
            )

            match avg_method:
                case "geometric":
                    _small_d = mp.sqrt(_shratio_12 * _shratio_21)
                case "distance":
                    _small_d = mp.sqrt(mp.fdiv((_shratio_12**2 + _shratio_21**2), "2"))
                case _:
                    _small_d = mp.fdiv(_shratio_12 + _shratio_21, "2")

            if _small_d < _delta_star:
                _s_2 += _gbd_step_sz
            else:
                break

        _gbdry_points.append((_s_1, _s_2))

        _s_2_oddsum += _s_2 if _s_2_oddval else 0
        _s_2_evnsum += _s_2 if not _s_2_oddval else 0
        _s_2_oddval = not _s_2_oddval

    # Starting at _s_id - _gbd_step_sz means _s_1 is not always
    # an even multiple of _gbd_step_sz
    _s_incpt = _s_2

    _gbd_prtlarea = 2 * _gbd_step_sz * (
        mp.fmul(4 / 3, _s_2_oddsum)
        + mp.fmul(2 / 3, _s_2_evnsum)
        + mp.fmul(1 / 3, _s_mid + _s_incpt)
    ) - mp.power(_s_mid, 2)

    _gbdry_points = np.array(_gbdry_points, np.float_)
    return (
        np.row_stack((np.flip(_gbdry_points, 0), np.flip(_gbdry_points[1:], 1))),
        round(float(_gbd_prtlarea), gbd_dps),
    )
