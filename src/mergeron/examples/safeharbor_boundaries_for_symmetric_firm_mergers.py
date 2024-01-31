"""
Draw GUPPI safe harbor boundary for symmetric firm mergers

This boundary is limited to share-margin space, given restrictions from symmetry
"""

from pathlib import Path

import numpy as np
from matplotlib import cm as colormgr
from matplotlib import colors as mcolors
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, StrMethodFormatter

import mergeron.core.guidelines_standards as gsf

prog_path = Path(__file__)
data_path = Path.home() / prog_path.parents[1].stem
if not data_path.is_dir():
    data_path.mkdir(parents=True)


def main() -> None:
    plt, _, _, set_axis_def = gsf.boundary_plot()
    # plt.delaxes(ax1)
    # del my_fig1, ax1
    # del ax1

    pcm_colornorm = mcolors.Normalize(0, 1.0)
    cmap_kwargs = {"cmap": "cividis", "norm": pcm_colornorm}

    my_fig1 = plt.figure(figsize=(5.5, 5.0), dpi=600)

    fig1_grid = my_fig1.add_gridspec(
        nrows=1,
        ncols=2,
        figure=my_fig1,
        width_ratios=[5, 0.5],
        height_ratios=[1.0],
        wspace=0.0,
    )
    ax1 = my_fig1.add_subplot(fig1_grid[0, 0])
    ax1 = set_axis_def(ax1, share_axes_flag=False)

    print("Generate data for plots")
    # Mgn coords, and div-ratio coords
    dh_bar, g_bar, r_bar, dr_bar = 0.01, 0.06, 0.80, 0.20

    step_size = 10**-5
    sym_shr_vec = np.arange(gsf.shr_from_gbd(g_bar), 0.5 + step_size, step_size)
    mst_vec = (g_bar / r_bar) * (1 - sym_shr_vec) / sym_shr_vec
    mst_vec_gsf = np.where(
        sym_shr_vec > gsf.shr_from_gbd(dr_bar, m_star=1.00), np.nan, mst_vec
    )

    print("Setup basic figure and axes for plots of safe harbor boundaries.")

    # Diversion-ratio boundary, in calibrated 6 percent safe-harbor
    ax1.plot(
        [np.sqrt(0.01 / 2) for _ in range(2)],
        [0.0, 1.0],
        label=r"$\Delta HHI = $100 points",
        linestyle="-",
        linewidth=0.5,
        color="black",
        zorder=5,
    )
    ax1.fill_between(
        [0.0, np.sqrt(0.01 / 2)],
        (1.0, 1.0),
        0,
        edgecolor=None,
        facecolor="#64bb64",
        alpha=0.7,
        zorder=2,
    )

    # Proposed GUPPI safe harbor
    ax1.scatter(
        sym_shr_vec,
        mst_vec,
        label=r"$\overline{g} = $6%; $\overline{d} = $20%",
        marker=",",
        s=(0.25 * 72.0 / my_fig1.dpi) ** 2,
        edgecolor=None,
        c=mst_vec,
        **cmap_kwargs,
        rasterized=True,
        zorder=9,
    )
    ax1.scatter(
        sym_shr_vec,
        0 * mst_vec,
        label=r"Projection, $\overline{g} = $6%; $\overline{d} = $20%",
        marker="o",
        s=(2.0 * 72.0 / my_fig1.dpi) ** 2,
        edgecolor=None,
        c=mst_vec,
        **cmap_kwargs,
        rasterized=True,
        zorder=11,
    )

    # Drop in a point for the Farrell/Shapiro example
    # Generate the points and error bars
    dr_bar_mgn = np.array([g_bar / d for d in (dr_bar, r_bar)])
    dr_bar_shr = np.round((dst_vec := g_bar / (r_bar * dr_bar_mgn)) / (1 + dst_vec), 3)
    ax1.scatter(
        dr_bar_shr,
        dr_bar_mgn,
        s=(5.0 * 72.0 / my_fig1.dpi) ** 2,
        marker=".",
        edgecolor=None,
        c=dr_bar_mgn,
        **cmap_kwargs,
    )
    ax1.plot(
        [dr_bar_shr[0] for _ in range(2)],
        [0, dr_bar_mgn[0]],
        label=r"$\overline{d} = $20%",
        linestyle="-",
        linewidth=0.75,
        alpha=1,
        color=colormgr.cividis(dr_bar_mgn[0]),
        zorder=5.5,
    )

    # Annotate diversion ratio limit
    ax1.annotate(
        r"$\overline{d}$ = 20%",
        xy=(0.20, 0.30),
        xytext=(0.20, 0.30 - 0.03),  # (0.20 + 0.005, 0.30 + 0.005),
        horizontalalignment="right",
        fontsize=8,
        zorder=5,
    )
    # Annotate diversion ratio limit
    ax1.annotate(
        r"$\overline{d}$ = 80%",
        xy=(0.50, 0.075),
        xytext=(0.5, 0.075 - 0.03),
        fontsize=8,
        horizontalalignment="right",
        zorder=5,
    )
    # Annotate \Deltah{} boundary
    ax1.annotate(
        r"$\Delta HHI$ = 100 points",
        xy=(0.07, 0.250),
        xytext=(0.07 + 0.005, 0.40 + 0.01),
        fontsize=8,
        rotation=90,
        zorder=5,
    )
    # Annotate GUPPI boundary
    ax1.annotate(
        r"$\overline{g}$ = 6%",
        xy=(0.10, 0.70),
        xytext=(0.10 + 0.004, 0.70 + 0.01),
        fontsize=8,
        zorder=5,
    )

    # Axis scale and labels
    # x-axis
    ax1.set_xlim(0.0, 0.5)
    ax1.set_xlabel("Symmetric Firm Share, $s$", fontsize=10)
    ax1.xaxis.set_label_coords(0.75, -0.1)
    # y-axis
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Price-Cost Margin, $m^*$", fontsize=10)
    ax1.yaxis.set_label_coords(-0.1, 0.75)

    _minorLocator = AutoMinorLocator(5)
    _majorLocator = MultipleLocator(0.05)
    for _axs in ax1.xaxis, ax1.yaxis:
        if _axs == ax1.xaxis:
            _majorticklabels_rot = 45
        elif _axs == ax1.yaxis:
            _majorticklabels_rot = 0
        # x-axis
        _axs.set_major_locator(_majorLocator)
        _axs.set_minor_locator(_minorLocator)
        _axs.set_major_formatter(StrMethodFormatter("{x:>3.0%}"))

        plt.setp(_axs.get_majorticklabels(), fontsize=6)
        plt.setp(_axs.get_majorticklabels(), rotation=_majorticklabels_rot)

    for axl in ax1.get_xticklabels(), ax1.get_yticklabels():
        plt.setp(axl[::2], visible=False)

    # Colorbar
    ax1_cb = my_fig1.add_subplot(fig1_grid[0, 1], frameon=False)
    ax1_cb.axis("off")
    cm_plot = my_fig1.colorbar(
        colormgr.ScalarMappable(**cmap_kwargs),
        use_gridspec=True,
        ax=ax1_cb,
        orientation="vertical",
        fraction=0.75,
        ticks=np.arange(0, 1.5, 0.1),
        format=StrMethodFormatter("{x:>3.0%}"),
        pad=0.0,
    )
    cm_plot.set_label(label="Price-Cost Margin", fontsize=8, fontweight=300)
    cm_plot.ax.tick_params(length=5, width=0.5, labelsize=6)
    cm_plot.ax.set_ylim(0, 1.0)
    cm_plot.outline.set_visible(False)

    plt.savefig(data_path.joinpath(f"{prog_path.stem}.pdf"))


if __name__ == "__main__":
    main()
