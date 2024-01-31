"""

    Defining the merging firm with the larger share as buyer under
    a GUPPI safeharbor bound of 6%, and the merging firm with the *smaller* share
    as the seller under a GUPPI safeharbor bound of 5%, and iteratively
    incrementing shares and margins by 5%, plot mergers that clear
    the safeharbor threshold, color-coded by margin of the firm with
    the larger GUPPI estimate.

"""

import contextlib
from pathlib import Path
from typing import BinaryIO

import numpy as np
import tables as tb
from matplotlib import cm as colormgr
from matplotlib import colors as mcolors
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
from numpy.typing import NDArray

import mergeron.core.guidelines_standards as gsf
import mergeron.gen.data_generation as dgl

prog_path = Path(__file__)
data_path = Path.home() / prog_path.parents[1].stem


g_bar = 0.06
r_bar = 0.80

sample_sz = 10**7
qtyshr_dist_type = "Uniform"
pcm_dist_type = "Uniform"

base_path = data_path / "{}_{}.zip".format(
    prog_path.stem, f"gbar{g_bar * 100:02.0f}PCT"
)

save_data_to_file_flag = False

h5path = data_path / f"{prog_path.stem}.h5"
blosc_filters = tb.Filters(complevel=3, complib="blosc:lz4", fletcher32=True)
h5datafile = tb.open_file(
    h5path, mode="w", title="Datasets, Sound GUPPI Safeharbor", filters=blosc_filters
)


def gen_plot_data(
    qtyshr_array: NDArray[float],
    divr_array: NDArray[float],
    qtyshr_firm1: NDArray[float],
    qtyshr_firm2: NDArray[float],
    _pcm_same_flag: bool = False,
    _pcm_firm2_star: float = 0.30,
    _guppi_aggr_method: str = "Maximum",
    /,
    *,
    h5handle: BinaryIO = h5datafile,
) -> tuple:
    assert _guppi_aggr_method in ("Maximum", "Average", "Minimum")

    h5hier = f"/plotData_pcmSame{_pcm_same_flag}"

    _pcm_firm1 = np.zeros((len(qtyshr_array), 1))
    _pcm_rng = dgl.MultithreadedRNG(_pcm_firm1, rng_dist_type=pcm_dist_type)
    _pcm_rng.fill()
    del _pcm_rng
    if _pcm_same_flag:
        _pcm_array = _pcm_firm1[:, [0, 0]]
    else:
        _pcm_array = np.column_stack((
            _pcm_firm1,
            _pcm_firm2_star * np.ones_like(_pcm_firm1),
        ))
    del _pcm_firm1

    _guppi_array = np.einsum("ij,ij->ij", _pcm_array[:, ::-1], divr_array)
    _stat_args = {"axis": 1, "keepdims": True}
    match _guppi_aggr_method:
        case "Minimum":
            _guppi_est = _guppi_array.min(**_stat_args)
            _gbd_test = (
                _guppi_est >= g_bar
            )  # (divr_array.max(axis=1, keepdims=True) < 0.2)
        case "Average":
            _guppi_est = _guppi_array.mean(**_stat_args)
            _gbd_test = _guppi_est < g_bar
        case _:
            _guppi_est = _guppi_array.max(**_stat_args)
            _gbd_test = _guppi_est < g_bar
    del _stat_args

    if _pcm_firm2_star >= 0.3 and _guppi_aggr_method == "Maximum":
        assert np.array_equal(
            _gbd_test[_gbd_test & (divr_array[:, [0]] < 0.2)],
            _gbd_test[_gbd_test & (_pcm_array[:, [1]] >= 0.3)],
        )

    _qtyshr_firm1_safe = qtyshr_firm1[_gbd_test]
    _qtyshr_firm2_safe = qtyshr_firm2[_gbd_test]
    _pcm_firm1_safe = _pcm_array[:, [0]][_gbd_test]
    _pcm_firm2_safe = _pcm_array[:, [1]][_gbd_test]
    del _gbd_test

    _pcm_plotter = _pcm_firm1_safe

    if save_data_to_file_flag:
        print("Save data to tables")
        for _array_name in (
            "qtyshr_firm1_safe",
            "qtyshr_firm2_safe",
            "pcm_firm1_safe",
            "pcm_firm2_safe",
        ):
            contextlib.suppress(tb.NoSuchNodeError)
            h5handle.remove_node(h5hier, name=_array_name)

            _array_h5 = h5handle.create_carray(
                h5hier,
                _array_name,
                obj=locals().get("_" + _array_name),
                createparents=True,
                title=f"{_array_name}",
            )
            _array_h5[:] = locals().get("_" + _array_name)

    if _guppi_aggr_method == "Minimum":
        _pcm_sorter = np.argsort(_pcm_plotter, axis=0)[::-1]
    else:
        _pcm_sorter = np.argsort(_pcm_plotter, axis=0)

    _qtyshr_firm1_plotter = _qtyshr_firm1_safe[_pcm_sorter]
    _qtyshr_firm2_plotter = _qtyshr_firm2_safe[_pcm_sorter]
    _pcm_plotter = _pcm_plotter[_pcm_sorter]
    del (
        _qtyshr_firm1_safe,
        _qtyshr_firm2_safe,
        _pcm_firm1_safe,
        _pcm_firm2_safe,
        _pcm_sorter,
    )

    return _qtyshr_firm1_plotter, _qtyshr_firm2_plotter, _pcm_plotter


# Generate a set of draws on shares and construct diversion ratios
def main():
    qtyshr_array = dgl.gen_market_shares_uniform(sample_sz).mktshr_array[:, :2]
    divr_array = r_bar * qtyshr_array[:, ::-1] / (1 - qtyshr_array)
    qtyshr_firm1, qtyshr_firm2 = np.split(qtyshr_array, 2, axis=1)

    fig_norm = mcolors.Normalize(0.0, 1.0)
    cmap_kwargs = {"cmap": "cividis", "norm": fig_norm}

    # Set up a plot grid to fill in the various scatterplots
    print(
        "Construct panel of scatter plots of cleared mergers",
        "by Firm 2 margin,",
        "color-coded by Firm 1 margin",
        sep=" ",
    )
    plt, my_fig1, ax1, set_axis_def = gsf.boundary_plot()
    plt.delaxes(ax1)
    del my_fig1, ax1

    my_fig_2dsg = plt.figure(figsize=(8.5, 9.5), dpi=600)

    fig_grid = my_fig_2dsg.add_gridspec(
        nrows=1,
        ncols=2,
        figure=my_fig_2dsg,
        height_ratios=[1.0],
        width_ratios=[6, 0.125],
        wspace=0.0,
    )
    fig_grid_gsf = fig_grid[0, 0].subgridspec(nrows=3, ncols=1, wspace=0, hspace=0.125)

    my_fig_3dsc = plt.figure(figsize=(5, 5), dpi=600)

    for ax_col, ax_collbl in enumerate(("Maximum", "Average", "Minimum")[:1]):
        for ax_row, ax_rowtup in enumerate((
            (False, 1.00),
            (False, 0.3),
            (False, g_bar / r_bar),
        )):
            pcm_same_flag, pcm_firm2_star = ax_rowtup
            ax_now = my_fig_2dsg.add_subplot(fig_grid_gsf[ax_row, ax_col])
            ax_now = set_axis_def(plt, ax_now)
            ax_now.set_xlabel(None)
            ax_now.set_ylabel(None)
            plt.setp(ax_now.get_xticklabels()[1::2], visible=False)
            plt.setp(ax_now.get_yticklabels()[1::2], visible=False)

            pcm_firm2_star_pct = pcm_firm2_star * 100
            # qtyshr_label
            ax_now.text(
                0.81,
                0.72,
                "\n".join([
                    r"$m_2 = m^* = {0:.{1}f}\%$".format(
                        pcm_firm2_star_pct, 1 * (pcm_firm2_star_pct % 1 > 0)
                    ),
                    r"$m_1 \neq m_2$",
                ]),
                rotation=0,
                ha="right",
                va="top",
                fontsize=10,
                fontweight=300,
                zorder=5,
            )
            del pcm_firm2_star_pct

            if ax_col == 0:
                if ax_row == 0:
                    # Set y-axis label
                    ax_now.yaxis.set_label_coords(-0.20, 1.0)
                    ax_now.set_ylabel(
                        "Firm 2 Market Share, $s_2$",
                        rotation=90,
                        ha="right",
                        va="top",
                        fontsize=10,
                        fontweight=300,
                    )
                elif ax_row == 2:
                    ax_now.xaxis.set_label_coords(1.0, -0.15)
                    ax_now.set_xlabel(
                        "\n".join(["Firm 1 Market Share, $s_1$"]),
                        ha="right",
                        fontsize=10,
                        fontweight=300,
                    )
                else:
                    pass

            qtyshr_firm1_plotter, qtyshr_firm2_plotter, pcm_plotter = gen_plot_data(
                qtyshr_array,
                divr_array,
                qtyshr_firm1,
                qtyshr_firm2,
                pcm_same_flag,
                pcm_firm2_star,
                ax_collbl,
                h5handle=h5datafile,
            )

            # # # GUPPI boundary for symmetric firm mergers
            # step_size = 10 ** -5
            # mst_vec = np.arange(g_bar / r_bar, 1.00 + step_size, step_size)
            # sym_shr_vec = (dst_vec := g_bar / (r_bar * mst_vec)) / (1 + dst_vec)
            # ax_now.scatter(
            #         sym_shr_vec, sym_shr_vec,
            #         marker='.', s=(0.1 * 72.0 / fig_2dsg.dpi) ** 2,
            #         c=mst_vec, **cmap_kwargs,
            #         rasterized=True,
            #         zorder=11,
            #         )

            ax_now.scatter(
                qtyshr_firm1_plotter,
                qtyshr_firm2_plotter,
                marker=".",
                s=(0.1 * 72.0 / my_fig_2dsg.dpi) ** 2,
                c=pcm_plotter,
                **cmap_kwargs,
                rasterized=True,
            )
            ax_now.set_aspect(1.0)

            if ax_collbl == "Maximum":
                ax3 = my_fig_3dsc.add_subplot(projection="3d")
                ax3.scatter(
                    qtyshr_firm1_plotter,
                    qtyshr_firm2_plotter,
                    pcm_plotter,
                    marker=",",
                    s=1,
                    c=pcm_plotter,
                    **cmap_kwargs,
                    rasterized=True,
                )

                for setlim in ax3.set_xlim, ax3.set_ylim, ax3.set_zlim:
                    setlim(0, 1)

                _majorLocator = MultipleLocator(0.2)
                for axidx, setaxis in enumerate((ax3.xaxis, ax3.yaxis, ax3.zaxis)):
                    setaxis.set_major_locator(_majorLocator)
                    setaxis.set_major_formatter(StrMethodFormatter("{x:3.0%}"))
                    setaxis.set_rotate_label(False)
                    plt.setp(
                        setaxis.get_majorticklabels(),
                        horizontalalignment=("right" if axidx else "left"),
                    )
                    setaxis.labelpad = 7
                ax3.set_xlabel(
                    "Firm 1 Market Share, $s_1$",
                    rotation=33.0,
                    fontsize=10,
                    fontweight=300,
                )
                ax3.set_ylabel(
                    "Firm 2 Market Share, $s_2$",
                    rotation=-33.0,
                    fontsize=10,
                    fontweight=300,
                )
                ax3.set_zlabel(
                    "Price-Cost Margin", rotation=-90.0, fontsize=10, fontweight=300
                )
                for axidx, axissel in enumerate(("x", "y", "z")):
                    ax3.tick_params(
                        axis=axissel,
                        which="major",
                        direction="inout",
                        length=5,
                        labelsize=6.0,
                        labelrotation=(-33.0, 33.0, 0.0)[axidx],
                        pad=-0.5,
                    )

                cm_3d = my_fig_3dsc.colorbar(
                    colormgr.ScalarMappable(**cmap_kwargs),
                    ax=ax3,
                    orientation="vertical",
                    shrink=0.35,
                    ticks=np.arange(0, 1.2, 0.2),
                    format=StrMethodFormatter("{x:<3.0%}"),
                )
                cm_3d.ax.tick_params(length=5, labelsize=6)  # width=0.25,
                cm_3d.outline.set_visible(False)
                plt.setp(
                    cm_3d.ax.yaxis.get_majorticklabels(),
                    horizontalalignment="left",
                    fontsize=6,
                    fontweight=300,
                )

                ax3.view_init(elev=30.0, azim=45.0)

                my_fig_3dsc_savepath = base_path.parent.joinpath(
                    "{}_{}_{}_{}_3dScatter.pdf".format(
                        base_path.stem,
                        f"pcmSame{pcm_same_flag}",
                        f"pcmStar{pcm_firm2_star * 100:.0f}",
                        f"gAgg{ax_collbl[:3].upper()}",
                    )
                )
                my_fig_3dsc.savefig(my_fig_3dsc_savepath, dpi=600)
                plt.close(my_fig_3dsc)
    h5datafile.close()

    # Colorbar
    ax_cm = my_fig_2dsg.add_subplot(fig_grid[-1, -1], frameon=False)
    #  https://stackoverflow.com/questions/14908576/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frame
    ax_cm.axis("off")
    #  https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.colorbar.html
    cm_plot = my_fig_2dsg.colorbar(
        colormgr.ScalarMappable(**cmap_kwargs),
        use_gridspec=True,
        ax=ax_cm,
        orientation="vertical",
        fraction=3.0,
        ticks=np.arange(0, 1.2, 0.2),
        format=StrMethodFormatter("{x:>3.0%}"),
    )
    cm_plot.set_label(
        label="Firm 1 Price-Cost Margin, $m_1$", fontsize=10, fontweight=300
    )
    cm_plot.ax.tick_params(length=5, width=0.5, labelsize=6)
    plt.setp(
        cm_plot.ax.yaxis.get_majorticklabels(),
        horizontalalignment="left",
        fontsize=6,
        fontweight=300,
    )

    #  https://stackoverflow.com/questions/27663136/remove-colorbars-borders-matplotlib/27672236
    # cm_plot.outline.set_linewidth(0.5)
    cm_plot.outline.set_visible(False)

    my_fig_2dsg_savepath = base_path.parent / f"{base_path.stem}_2DScatterGrid.pdf"
    print(f"Save 2D plot to, {my_fig_2dsg_savepath!r}")
    my_fig_2dsg.savefig(my_fig_2dsg_savepath, dpi=600)


if __name__ == "__main__":
    main()
