"""

Draw boundaries for various standards from U.S. Horizontal Merger Guidelines.
"""

from itertools import product as iterprod
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib.axis
from matplotlib import cm as colormgr
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.ticker import StrMethodFormatter
from numpy import arange, arctan, array, hsplit, insert, rad2deg, round, sqrt, vstack

import mergeron.core.guidelines_boundaries as gbl
from mergeron import DATA_DIR, UPPAggrSelector
from mergeron.core import UPPBoundarySpec

PROG_PATH = Path(__file__)


class _CMAPArgs(TypedDict):
    cmap: mcolors.Colormap
    norm: mcolors.Normalize


_color_kwargs = _CMAPArgs(cmap=colormaps["cividis"], norm=mcolors.Normalize(0, 1.0))


def plot_delta_boundaries(
    _guppi_bench_key: str,
    _print_guppi_max_bndry_envs_flag: bool,
    _recapture_spec: Literal["inside-out", "proportional"],
    _color_kwargs: _CMAPArgs = _color_kwargs,
    /,
) -> None:
    _print_guppi_max_bndry_envs_flag = _print_guppi_max_bndry_envs_flag or False

    print("ΔHHI safeharbor boundary")
    _plt, _my_fig1, _ax1, _ = gbl.boundary_plot()

    _hmg_thresholds = get_hmg_thresholds_by_key(_guppi_bench_key)
    _dh_bar, _r_bar, _guppi_bench, _divr_bench = (
        getattr(_hmg_thresholds, _f) for _f in ("delta", "rec", "guppi", "divr")
    )

    print("Contour map of selected ∆HHI boundaries")

    for _dh_bound in 100, 200, 300, 500, 800, 1200, 2500:
        if _dh_bound in (300, 500):
            continue

        _dh_boundary = gbl.delta_hhi_boundary(_dh_bound / 1e4)
        _dh_dat_x, _dh_dat_y = (_z.T[0] for _z in hsplit(_dh_boundary.coordinates, 2))
        if _dh_bound == 100:
            _lwval, _lsval = 0.75, "-"
        else:
            _lwval, _lsval = 0.5, ":"

        _ax1.plot(
            _dh_dat_x,
            _dh_dat_y,
            linewidth=_lwval,
            linestyle=_lsval,
            color="black",
            zorder=3,
        )

        if _print_guppi_max_bndry_envs_flag:
            _symshr = gbl.round_cust(sqrt(_dh_bound / 2e4))
            _dstar = gbl.round_cust(_symshr / (1 - _symshr))
            _m_star = _guppi_bench / (_dstar * _r_bar)
            print(_symshr, _dstar, _m_star, _r_bar, _guppi_bench, "...", end="")

            _guppi_bdry_env_xs = (0, _symshr, 1)
            if _recapture_spec == "inside-out":
                # ## Plot envelope of GUPPI boundaries with
                #   r_k = r_bar if s_k = min(_s_1, _s_2)
                # ## See (s_i, s_j) in equation~(44), or thereabouts, in paper
                _smin_nr = _dstar * (1 - _r_bar)
                _smax_nr = 1 - _dstar * _r_bar
                _guppi_bdry_env_dr = _smin_nr + _smax_nr
                _guppi_bdry_env_xs = (  # type: ignore
                    0,
                    _smin_nr / _guppi_bdry_env_dr,
                    _symshr,
                    _smax_nr / _guppi_bdry_env_dr,
                    1,
                )
                del _guppi_bdry_env_dr
                print(_guppi_bdry_env_xs[1], _guppi_bdry_env_xs[-2], end="")
            print()

            _ax1.plot(
                _guppi_bdry_env_xs,
                _guppi_bdry_env_xs[::-1],
                linewidth=0.5,
                linestyle="--",
                color=_color_kwargs["cmap"](_m_star),
                zorder=3,
            )
            del _symshr, _dstar, _m_star

            if _recapture_spec == "inside-out" and _dh_bound not in (200, 300, 500):
                _ax1.annotate(
                    rf"$\Delta HHI$ = {_dh_bound:,d} pts.",
                    xy=(_dh_dat_x[1], _dh_dat_y[1]),
                    xytext=(_dh_dat_x[1], _dh_dat_y[1]),
                    textcoords="data",
                    ha="left",
                    va="center",
                    fontsize=5,
                    zorder=5.1,
                )
            del _guppi_bdry_env_xs

        if _dh_bound == 100:
            _ax1.fill_between(
                _dh_dat_x,
                _dh_dat_y,
                0,
                edgecolor=None,
                facecolor="#64bb64",
                alpha=0.7,
                rasterized=True,
            )
        del (_dh_boundary, _dh_dat_x, _dh_dat_y)  # , _dh_dat_x_pla, _dh_dat_y_pla

    _my_fig1.savefig((DATA_DIR / f"{PROG_PATH.stem}_DH100_deltaHHI_only.pdf"), dpi=600)

    del _plt, _my_fig1, _ax1


def plot_guppi_boundaries(  # noqa PLR0915
    _guppi_bench_key: str,
    _pcm_same_flag: bool,
    _recapture_spec: Literal["proportional", "inside-out"],
    _color_kwargs: _CMAPArgs = _color_kwargs,
    /,
) -> None:
    if recapture_spec not in (_recspecs := ("inside-out", "proportional")):
        raise ValueError(f"Recapture specification must be one of, {_recspecs!r}")

    _hmg_thresholds = get_hmg_thresholds_by_key(_guppi_bench_key)
    _dh_bar, _r_bar, _guppi_bench, _divr_bench = (
        getattr(_hmg_thresholds, _f) for _f in ("delta", "rec", "guppi", "divr")
    )

    # First we get the data for the ΔHHI benchmark we want to plot
    _dh_boundary = gbl.delta_hhi_boundary(_dh_bar)
    _dh_dat_x, _dh_dat_y = (_z.T[0] for _z in hsplit(_dh_boundary.coordinates, 2))

    _plt, _, _, _set_axis_def = gbl.boundary_plot()

    _my_fig1 = _plt.figure(figsize=(5.5, 5.0))

    _fig1_grid = _my_fig1.add_gridspec(
        nrows=1,
        ncols=2,
        figure=_my_fig1,
        width_ratios=[5, 0.5],
        height_ratios=[1.0],
        wspace=0.0,
    )
    _ax1 = _my_fig1.add_subplot(_fig1_grid[0, 0])
    _ax1 = _set_axis_def(_ax1, mktshares_plot_flag=True, mktshares_axlbls_flag=True)

    _ax1.plot(_dh_dat_x, _dh_dat_y, linewidth=0.75, color="black", zorder=3)
    _ax1.fill_between(
        _dh_dat_x,
        _dh_dat_y,
        0,
        edgecolor=None,
        facecolor="#64bb64",
        alpha=0.7,
        rasterized=True,
    )

    # GUPPI boundary-of-boundaries for symmetric-firm mergers
    _step_size = 10**-5
    _m_lim = _guppi_bench / _r_bar
    _mst_vec = arange(_m_lim, 1.00 + _step_size, _step_size)
    _sym_shr_vec = (_dst_vec := _m_lim / _mst_vec) / (1 + _dst_vec)
    # _sym_shr_vec = np.arange(gbl.shr_from_gbd(g_bar), 0.5 + step_size, step_size)
    # _mst_vec = (g_bar / r_bar) * (1 - sym_shr_vec) / sym_shr_vec
    # https://stackoverflow.com/questions/39753282/
    _ax1.scatter(
        _sym_shr_vec,
        _sym_shr_vec,
        label=r"$\overline{g}$ = 6%; $\overline{d}$ = 20%",
        marker=",",
        s=(0.25 * 72.0 / _my_fig1.dpi) ** 2,
        edgecolor=None,
        c=_mst_vec,
        **_color_kwargs,
        rasterized=True,
        zorder=3,
    )

    for _m_star in arange(1.00, _m_lim - 0.10, -0.10):
        _m_star, _delta_star, _s_mid = (
            round(_s, 4)
            for _s in (_m_star, _m_lim / _m_star, _m_lim / (_m_star + _m_lim))
        )

        _s2_vals = (_delta_star, _s_mid, 0.0)
        if _pcm_same_flag:
            _s1_vals = _s2_vals[::-1]
            _ax1.plot(
                _s1_vals,
                _s2_vals,
                linestyle="--",
                linewidth=0.75,
                color=_color_kwargs["cmap"](_m_star),
                zorder=3,
            )
        else:
            _s1_vals = (0.0, _s_mid, 1.0)
            if _recapture_spec == "inside-out":
                _s_1_i2, _s_2_i2 = array([
                    (_x := _guppi_bench - _m_star),
                    (_y := _guppi_bench - _m_lim),
                ]) / (_x + _y)
                del _x, _y

                _s1_vals = insert(_s1_vals, [-1], [_s_1_i2])
                _s2_vals = insert(_s2_vals, [-1], [_s_2_i2])

            # Print Outer boundary, color-coded m_2
            _ax1.plot(
                _s1_vals,
                _s2_vals,
                linestyle="--",
                linewidth=0.5,
                color=colormaps["cividis"](_m_star),
                zorder=3,
            )
            # Print inner boundary, color-coded m_1
            _ax1.plot(
                _s2_vals[::-1],
                _s1_vals[::-1],
                linestyle="--",
                linewidth=0.5,
                color=colormaps["cividis"](_m_star),
                zorder=3,
            )

        # skip labeling a few
        if _m_star >= 0.70:
            continue

        # Place boundary labels along each boundary-line segment
        _bndry_grad = grad_est(_ax1, (0, _s_mid), (_delta_star, _s_mid))
        _bndry_angle = rad2deg(arctan(_bndry_grad))

        _x_shft_pts = 0.001
        _y_shft_pts = 0.015

        _m1_lbl_str, _m2_lbl_str = (
            r"${0} = {1:3.{2}f}\%$".format(
                ("m^*" if _pcm_same_flag else f"m_{_k}"),
                (_pcmv := _m_star * 100),
                2 * int(_pcmv % 1 > 0),
            )
            for _k in (1, 2)
        )
        _ax1.annotate(
            _m2_lbl_str,
            xy=(0, _delta_star),
            xytext=(0 + _x_shft_pts, _delta_star + _y_shft_pts),
            textcoords="data",
            rotation=_bndry_angle,
            ha="left",
            va="top",
            fontsize=4,
            zorder=5.1,
        )
        _ax1.annotate(
            _m1_lbl_str,
            xy=(_delta_star, 0),
            xytext=(0 + _x_shft_pts, _delta_star + _y_shft_pts)[::-1],
            textcoords="data",
            rotation=-90 - _bndry_angle,
            ha="right",
            va="bottom",
            fontsize=4,
            zorder=5.1,
        )

    # Examples of hypothetical combinations
    if not _pcm_same_flag:
        if _guppi_bench_key == "DH50":
            _xco, _yco, _lval = zip(
                *([0.15, 0.10, "A"], [0.27, 0.15, "B"]), strict=True
            )
        else:
            _xco, _yco, _lval = zip(
                *([0.20, 0.135, "A"], [0.27, 0.15, "B"]), strict=True
            )

        _ax1.scatter(_xco, _yco, s=1, c="black", zorder=4.5)
        for lid, lvl in enumerate(_lval):
            _ax1.annotate(
                f"${lvl}$",
                xy=(_xco[lid], _yco[lid]),
                xytext=(_xco[lid] - 0.025, _yco[lid] - 0.01),
                fontsize=8,
                zorder=4.6,
            )

    #  print("Diversion ratio bound")
    if _divr_bench < _r_bar:
        _m_star_bench = _guppi_bench / _divr_bench
        _s_mid_bench = gbl.shr_from_gbd(
            _guppi_bench, m_star=_m_star_bench, r_bar=_r_bar
        )
        _delta_star = gbl.critical_shrratio(
            _guppi_bench, m_star=_m_star_bench, r_bar=_r_bar
        )
        guppi_boundary = gbl.shrratio_boundary(
            UPPBoundarySpec(_delta_star, _r_bar, agg_method=UPPAggrSelector.MAX)
        )
        _x_drt, _y_drt = zip(*guppi_boundary.coordinates, strict=True)

        _ax1.plot(
            _x_drt,
            _y_drt,
            linestyle="-",
            linewidth=0.75,
            color=_color_kwargs["cmap"](_m_star_bench),
            zorder=4,
        )

        _drt_grad = grad_est(_ax1, _x_drt[:2], _y_drt[:2])
        _drt_angle = rad2deg(arctan(_drt_grad))

        _dl_xshift = 0.015 if _divr_bench < 0.20 else -0.01
        _dl_yshift = 0.0075 if _divr_bench < 0.20 else 0.005

        _dr_labelstr = R"$\symbf{{\overline{{d}} = {0:3.{1}f}\%}}$".format(
            (_dbch := _divr_bench * 100), 1 * (_dbch % 1 > 0)
        )

        _ax1.annotate(
            _dr_labelstr,
            xy=(_s_mid_bench, _s_mid_bench),
            xytext=(_s_mid_bench + _dl_xshift, _s_mid_bench + _dl_yshift),
            rotation=_drt_angle,
            ha="right",
            va="bottom",
            fontsize=5,
            zorder=4.1,
        )
        _ax1.annotate(
            _dr_labelstr,
            xy=(_s_mid_bench, _s_mid_bench),
            xytext=(_s_mid_bench + _dl_xshift, _s_mid_bench + _dl_yshift)[::-1],
            textcoords="data",
            rotation=-90 - _drt_angle,
            ha="left",
            va="top",
            fontsize=5,
            zorder=4.1,
        )

        del _x_drt, _y_drt, _drt_grad, _drt_angle

    # Drop in a point for the Farrell/Shapiro example
    # Generate the points and error bars
    _ex_divratio = 1.0
    _ex_shr_from_divratio = round(_ex_divratio / (1 + _ex_divratio), 3)
    _ebar_array = vstack([_ex_shr_from_divratio, 0.0 * _ex_shr_from_divratio])
    _ebar_plot = _ax1.errorbar(
        _ex_shr_from_divratio,
        _ex_shr_from_divratio,
        xerr=_ebar_array,
        yerr=_ebar_array,
        fmt=".",
        mfc=_color_kwargs["cmap"](_guppi_bench / _r_bar),
        mec="None",
        alpha=0.9,
        zorder=5,
    )
    # Set linestyle for errorbars
    # https://stackoverflow.com/questions/22995797/
    for _ix in range(2):
        _ebar_plot[-1][_ix].set(
            color=_color_kwargs["cmap"](_guppi_bench / _r_bar),
            linestyle="--",
            linewidth=0.5,
            alpha=0.9,
            zorder=2,
        )

    # Annotate the point
    _ax1.annotate(
        rf"$d={_r_bar * 100:3.0f}\%$",
        xy=(_ex_shr_from_divratio, _ex_shr_from_divratio),
        xytext=(_ex_shr_from_divratio + 0.01, _ex_shr_from_divratio - 0.008),
        fontsize=5,
        zorder=5.1,
    )

    # Colorbar
    ax1_cb = _my_fig1.add_subplot(_fig1_grid[0, 1], frameon=False)
    ax1_cb.axis("off")
    _cm_plot = _my_fig1.colorbar(
        colormgr.ScalarMappable(**_color_kwargs),
        use_gridspec=True,
        ax=ax1_cb,
        orientation="vertical",
        fraction=0.75,
        ticks=arange(0, 1.5, 0.1),
        format=StrMethodFormatter("{x:>3.0%}"),
        pad=0.0,
    )
    _cm_plot.set_label(label="Price-Cost Margin", fontsize=8)
    _cm_plot.ax.tick_params(length=5, width=0.5, labelsize=6)
    _cm_plot.ax.set_ylim(0, 1.0)
    _cm_plot.outline.set_visible(False)

    _fig1_savename = "_".join((
        f"{PROG_PATH.stem}",
        f"pcmSame{_pcm_same_flag}",
        f"{_guppi_bench_key}",
        f"rbar{_recapture_spec.upper()}",
    ))
    _my_fig1.savefig((DATA_DIR / f"{_fig1_savename}.pdf"), dpi=600)
    del _my_fig1, _ax1, _ebar_plot, _fig1_savename
    del _dh_dat_x, _dh_dat_y


def grad_est(
    _ax: matplotlib.axis.Axis, _pt_xs: tuple[float, ...], _pt_ys: tuple[float, ...]
) -> float:
    if (_pt_len := max(len(_pt_xs), len(_pt_ys))) > 2:
        raise ValueError(
            "Expecting only 2 points for calculation of line-gradient; got {_pt_len}."
        )
    _pt1, _pt2 = (
        _ax.transData.transform_point((_pt_xs[_i], _pt_ys[_i]))  # type: ignore
        for _i in range(2)
    )
    _grad: float = (_pt2[1] - _pt1[1]) / (_pt2[0] - _pt1[0])
    return _grad


def get_hmg_thresholds_by_key(_guppi_bench_key: str, /) -> gbl.HMGThresholds:
    match _guppi_bench_key:
        case "DOJATR":
            return gbl.HMGThresholds(
                (_tmp := gbl.GuidelinesThresholds(2010).safeharbor).delta,
                _tmp.rec,
                0.05,
                _tmp.divr,
                _tmp.cmcr,
                _tmp.ipr,
            )
        case "DH100":
            return gbl.GuidelinesThresholds(2010).safeharbor
        case "DH50":
            return gbl.GuidelinesThresholds(1992).safeharbor
        case _:
            raise ValueError(
                f"GUPPI benchmark key must be one of, {guppi_benchmark_keys!r}"
            )


if __name__ == "__main__":
    print("Define parameters for GUPPI safeharbor plots")

    guppi_benchmark_keys = ("DH50", "DH100", "DOJATR")

    print("Plot countour-maps of ∆HHI boundaries and GUPPI safeharbor boundaries")

    pcm_same_flag: bool = True
    recapture_spec: Literal["inside-out", "proportional"] = "inside-out"
    plot_delta_boundaries("DH100", pcm_same_flag, recapture_spec, _color_kwargs)

    print("GUPPI safeharbor boundaries, by precentage price-cost margin")
    for pcm_same_flag, recapture_spec in iterprod(
        (False, True), ("inside-out", "proportional")
    ):
        if pcm_same_flag and recapture_spec == "inside-out":
            continue

        print(f"Symmetric margins: {pcm_same_flag!r}; recapture: '{recapture_spec}'")
        for guppi_bench_key in guppi_benchmark_keys:
            plot_guppi_boundaries(
                guppi_bench_key, pcm_same_flag, recapture_spec, _color_kwargs
            )
