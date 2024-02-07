"""

    Defining the merging firm with the larger share as buyer under a
    GUPPI safeharbor bound of 6%, and the merging firm with the
    *smaller* share as the buyer under a GUPPI safeharbor bound of 5%,
    and incrementing shares and margins by 5%, plot mergers that
    clear the safeharbor threshold, color-coded by margin of the firm
    with the larger GUPPI estimate.

"""

from __future__ import annotations

import gc
from contextlib import suppress
from pathlib import Path
from typing import Final

import numpy as np
import tables as tb
from matplotlib import cm, colors
from matplotlib.ticker import StrMethodFormatter
from numpy import arange, argsort, column_stack, einsum, ones_like
from numpy.typing import NDArray

import mergeron.core.guidelines_standards as gsf
import mergeron.gen.data_generation as dgl
import mergeron.gen.investigations_stats as clstl
from mergeron.core.psuedorandom_numbers import DEFAULT_PARM_ARRAY

prog_path = Path(__file__)
data_path = Path.home() / prog_path.parents[1].stem


# Get Guidelines parameter values
hmg_pub_year: Final = 2023
r_bar, g_bar, divr_bar, *_ = gsf.GuidelinesStandards(hmg_pub_year).presumption[2:]

sample_sz = 10**7

market_sample_spec = dgl.MarketSampleSpec(
    sample_sz,
    r_bar,
    share_spec=(dgl.SHRConstants.UNI, dgl.RECConstants.INOUT, DEFAULT_PARM_ARRAY),
)

base_path = data_path / "{}_gbar{}PCT_{}Recapture.zip".format(
    prog_path.stem, f"{g_bar * 100:02.0f}", market_sample_spec.share_spec[1]
)
save_data_to_file_flag = False
h5path = data_path / f"{base_path.stem}.h5"
blosc_filters = tb.Filters(
    complevel=3, complib="blosc:lz4", bitshuffle=True, fletcher32=True
)
h5datafile = tb.open_file(
    h5path,
    mode="w",
    title="Datasets, Sound GUPPI Safeharbor, Envelopes of GUPPI Boundaries",
    filters=blosc_filters,
)


def gen_plot_data(
    _market_data: dgl.MarketSample,
    _pcm_firm2_star: float = 0.30,
    _clrenf_sel: clstl.CLRENFSelector = clstl.CLRENFSelector.CLRN,
    /,
    *,
    h5handle: tb.File = h5datafile,
) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
    h5hier = "/plotData_mstar{}PCT".format(
        f"{_pcm_firm2_star * 100:03.1f}".replace(".", "dot")
    )

    _pcm_array = column_stack((
        _m1 := _market_data.pcm_array[:, [0]],
        _pcm_firm2_star * ones_like(_m1),
    ))
    del _m1

    _guppi_array = einsum("ij,ij->ij", _pcm_array[:, ::-1], _market_data.divr_array)
    _guppi_est = _guppi_array.min(axis=1, keepdims=True)

    if _clrenf_sel == clstl.CLRENFSelector.CLRN:
        _gbd_test = _guppi_est < g_bar
    else:
        _gbd_test = _guppi_est >= g_bar

    _gbd_test_rows = np.where(_gbd_test)[0]

    _qtyshr_firm1_safe, _qtyshr_firm2_safe = (
        _market_data.frmshr_array[_gbd_test_rows][:, [0]],
        _market_data.frmshr_array[_gbd_test_rows][:, [1]],
    )
    _pcm_firm1_safe, _pcm_firm2_safe = (
        _pcm_array[_gbd_test_rows][:, [0]],
        _pcm_array[_gbd_test_rows][:, [1]],
    )
    del _gbd_test, _gbd_test_rows

    _pcm_plotter = _pcm_firm1_safe

    if save_data_to_file_flag:
        print("Save data to tables")
        for _array_name in (
            "qtyshr_firm1_safe",
            "qtyshr_firm2_safe",
            "pcm_firm1_safe",
            "pcm_firm2_safe",
        ):
            with suppress(tb.NoSuchNodeError):
                h5handle.remove_node(h5hier, name=_array_name)
            _array_h5 = h5handle.create_carray(
                h5hier,
                _array_name,
                obj=locals().get(f"_{_array_name}"),
                createparents=True,
                title=f"{_array_name}",
            )

    _pcm_sorter = argsort(_pcm_plotter, axis=0)
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


# Generate market data
def main(
    _market_sample_spec: dgl.MarketSampleSpec, _clrenf_sel: clstl.CLRENFSelector
) -> None:
    market_data = dgl.gen_market_sample(_market_sample_spec, seed_seq_list=None)

    # Set up a plot grid to fill in the various scatterplots
    print(
        "Construct panel of scatter plots of cleared mergers by Firm 2 margin",
        "with each plot color-coded by Firm 1 margin",
        sep=", ",
    )
    fig_norm = colors.Normalize(0.0, 1.0)
    cmap_kwargs = {"cmap": "cividis", "norm": fig_norm}
    plt, _, _, set_axis_def = gsf.boundary_plot()

    fig_2dsg = plt.figure(figsize=(8.5, 9.5), dpi=600)

    fig_grid = fig_2dsg.add_gridspec(
        nrows=1, ncols=2, figure=fig_2dsg, width_ratios=[6, 0.125], wspace=0.0
    )
    fig_grid_gsf = fig_grid[0, 0].subgridspec(nrows=3, ncols=1, wspace=0, hspace=0.125)

    # fig_3dsc = plt.figure(figsize=(5, 5), dpi=600)

    for ax_row, pcm_firm2_star in enumerate((1.00, g_bar / divr_bar, g_bar / r_bar)):
        ax_now = fig_2dsg.add_subplot(fig_grid_gsf[ax_row, 0])
        ax_now = set_axis_def(ax_now, share_axes_flag=True)
        ax_now.set_xlabel(None)
        ax_now.set_ylabel(None)
        # plt.setp(ax_now.get_xticklabels()[1::2], visible=False)
        # plt.setp(ax_now.get_yticklabels()[1::2], visible=False)

        ax_now.text(
            0.81,
            0.72,
            "\n".join([
                r"$m_2 = m^* = {0:.{1}f}\%$".format(
                    (_pcmv := pcm_firm2_star * 100), 1 * (_pcmv % 1 > 0)
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

        qtyshr_firm1_plotter, qtyshr_firm2_plotter, pcm_plotter = gen_plot_data(
            market_data, pcm_firm2_star, h5handle=h5datafile
        )

        ax_now.scatter(
            qtyshr_firm1_plotter,
            qtyshr_firm2_plotter,
            marker=".",
            s=(0.1 * 72.0 / fig_2dsg.dpi) ** 2,
            c=pcm_plotter,
            **cmap_kwargs,
            rasterized=True,
        )
        ax_now.set_aspect(1.0)

        # ax3 = fig_3dsc.add_subplot(projection="3d")
        # ax3.scatter(
        #     qtyshr_firm1_plotter,
        #     qtyshr_firm2_plotter,
        #     pcm_plotter,
        #     marker=",",
        #     s=1,
        #     c=pcm_plotter,
        #     **cmap_kwargs,
        #     rasterized=True,
        # )

        # for setlim in ax3.set_xlim, ax3.set_ylim, ax3.set_zlim:
        #     setlim(0, 1)

        # _majorLocator = MultipleLocator(0.2)
        # for axidx, setaxis in enumerate((ax3.xaxis, ax3.yaxis, ax3.zaxis)):
        #     setaxis.set_major_locator(_majorLocator)
        #     setaxis.set_major_formatter(StrMethodFormatter("{x:3.0%}"))
        #     setaxis.set_rotate_label(False)
        #     plt.setp(
        #         setaxis.get_majorticklabels(),
        #         horizontalalignment=("right" if axidx else "left"),
        #     )
        #     setaxis.labelpad = 7
        # ax3.set_xlabel(
        #     "Firm 1 Market Share, $s_1$", rotation=33.0, fontsize=10, fontweight=300
        # )
        # ax3.set_ylabel(
        #     "Firm 2 Market Share, $s_2$", rotation=-33.0, fontsize=10, fontweight=300
        # )
        # ax3.set_zlabel("Price-Cost Margin", rotation=-90.0, fontsize=10, fontweight=300)
        # for axidx, axissel in enumerate(("x", "y", "z")):
        #     ax3.tick_params(
        #         axis=axissel,
        #         which="major",
        #         direction="inout",
        #         length=5,
        #         labelsize=6.0,
        #         labelrotation=(-33.0, 33.0, 0.0)[axidx],
        #         pad=-0.5,
        #     )

        # cm_3d = fig_3dsc.colorbar(
        #     cm.ScalarMappable(**cmap_kwargs),
        #     ax=ax3,
        #     orientation="vertical",
        #     shrink=0.35,
        #     ticks=arange(0, 1.2, 0.2),
        #     format=StrMethodFormatter("{x:<3.0%}"),
        # )
        # cm_3d.ax.tick_params(length=5, labelsize=6)
        # cm_3d.outline.set_visible(False)
        # plt.setp(
        #     cm_3d.ax.yaxis.get_majorticklabels(),
        #     horizontalalignment="left",
        #     fontsize=6,
        #     fontweight=300,
        # )

        # ax3.view_init(elev=30.0, azim=45.0)

        # my_fig_3dsc_savepath = data_path / (
        #     f"{base_path.stem}_pcmStar{pcm_firm2_star * 100:.0f}_3dScatter.pdf"
        # )
        # fig_3dsc.savefig(my_fig_3dsc_savepath, dpi=600)
        # plt.close(fig_3dsc)
    h5datafile.close()
    gc.collect()

    # Colorbar
    ax_cm = fig_2dsg.add_subplot(fig_grid[-1, -1], frameon=False)
    ax_cm.axis("off")
    cm_plot = fig_2dsg.colorbar(
        cm.ScalarMappable(**cmap_kwargs),
        use_gridspec=True,
        ax=ax_cm,
        orientation="vertical",
        fraction=3.0,
        ticks=arange(0, 1.2, 0.2),
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

    cm_plot.outline.set_visible(False)

    my_fig_2dsg_savepath = base_path.parent / f"{base_path.stem}_2DScatterGrid.pdf"
    print(f"Save 2D plot to, {my_fig_2dsg_savepath!r}")
    fig_2dsg.savefig(my_fig_2dsg_savepath, dpi=600)


if __name__ == "__main__":
    main(market_sample_spec, clstl.CLRENFSelector.ENFT)

    h5datafile.close()
