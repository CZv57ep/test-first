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
import tables as ptb  # type: ignore
from matplotlib import cm, colors
from matplotlib.ticker import StrMethodFormatter
from numpy import arange, argsort, column_stack, einsum, ones_like
from numpy.typing import NDArray

import mergeron.core.guidelines_standards as gsf
import mergeron.gen.data_generation as dgl
import mergeron.gen.guidelines_tests as gtl
import mergeron.gen.investigations_stats as isl
from mergeron import DATA_DIR
from mergeron.core.pseudorandom_numbers import DIST_PARMS_DEFAULT

PROG_PATH = Path(__file__)


blosc_filters = ptb.Filters(
    complevel=3, complib="blosc:lz4", bitshuffle=True, fletcher32=True
)


def gen_plot_data(
    _market_data: dgl.MarketsSample,
    _g_bar: float,
    _pcm_firm2_star: float = 0.30,
    _inv_sel: gtl.UPPTestSpec = (
        isl.PolicySelector.CLRN,
        gtl.GUPPIWghtngSelector.MAX,
        None,
    ),
    /,
    *,
    h5handle: ptb.File | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    _h5hier = "/plotData_mstar{}PCT".format(
        f"{_pcm_firm2_star * 100:03.1f}".replace(".", "dot")
    )

    _pcm_array = column_stack((
        _m1 := _market_data.pcm_array[:, [0]],
        _pcm_firm2_star * ones_like(_m1),
    ))
    del _m1

    _guppi_array = einsum("ij,ij->ij", _pcm_array[:, ::-1], _market_data.divr_array)

    _guppi_est = (
        _guppi_array.max(axis=1, keepdims=True)
        if _inv_sel[1] == gtl.GUPPIWghtngSelector.MAX
        else _guppi_array.min(axis=1, keepdims=True)
    )

    _gbd_test = (
        (_guppi_est < _g_bar)
        if _inv_sel[0] == isl.PolicySelector.CLRN
        else (_guppi_est >= _g_bar)
    )

    del _guppi_est

    _gbd_test_rows = np.where(_gbd_test)[0]

    _qtyshr_firm1_inv, _qtyshr_firm2_inv = (
        _market_data.frmshr_array[_gbd_test_rows][:, [0]],
        _market_data.frmshr_array[_gbd_test_rows][:, [1]],
    )
    _pcm_firm1_inv, _pcm_firm2_inv = (
        _pcm_array[_gbd_test_rows][:, [0]],
        _pcm_array[_gbd_test_rows][:, [1]],
    )
    del _gbd_test, _gbd_test_rows

    _pcm_plotter = _pcm_firm1_inv

    if h5handle:
        print("Save data to tables")
        for _array_name in (
            "qtyshr_firm1_inv",
            "qtyshr_firm2_inv",
            "pcm_firm1_inv",
            "pcm_firm2_inv",
        ):
            with suppress(ptb.NoSuchNodeError):
                h5handle.remove_node(_h5hier, name=_array_name)
            _array_h5 = h5handle.create_carray(
                _h5hier,
                _array_name,
                obj=locals().get(f"_{_array_name}"),
                createparents=True,
                title=f"{_array_name}",
            )

    _pcm_sorter = argsort(_pcm_plotter, axis=0)
    if _inv_sel[0] != isl.PolicySelector.CLRN:
        _pcm_sorter = _pcm_sorter[::-1, :]
    _qtyshr_firm1_plotter = _qtyshr_firm1_inv[_pcm_sorter]
    _qtyshr_firm2_plotter = _qtyshr_firm2_inv[_pcm_sorter]
    _pcm_plotter = _pcm_plotter[_pcm_sorter]

    del (
        _qtyshr_firm1_inv,
        _qtyshr_firm2_inv,
        _pcm_firm1_inv,
        _pcm_firm2_inv,
        _pcm_sorter,
    )

    return _qtyshr_firm1_plotter, _qtyshr_firm2_plotter, _pcm_plotter


# Generate market data
def _main(
    _hmg_pub_year: gsf.HMGPubYear,
    _market_sample_spec: dgl.MarketSampleSpec,
    _inv_sel: gtl.UPPTestSpec,
    _save_data_to_file: gtl.SaveDataSpec,
) -> None:
    _r_bar, _g_bar, _divr_bar, *_ = getattr(
        gsf.GuidelinesStandards(_hmg_pub_year),
        "safeharbor" if _inv_sel[0] == isl.PolicySelector.ENFT else "presumption",
    )[2:]
    market_data = dgl.gen_market_sample(_market_sample_spec, seed_seq_list=None)

    # Set up a plot grid to fill in the various scatterplots
    print(
        "Construct panel of scatter plots of cleared mergers by Firm 2 margin",
        "with each plot color-coded by Firm 1 margin",
        sep=", ",
    )
    _fig_norm = colors.Normalize(0.0, 1.0)
    _cmap_kwargs = {"cmap": "cividis", "norm": _fig_norm}
    _plt, _, _, _set_axis_def = gsf.boundary_plot()

    _fig_2dsg = _plt.figure(figsize=(8.5, 9.5), dpi=600)

    _fig_grid = _fig_2dsg.add_gridspec(
        nrows=1, ncols=2, figure=_fig_2dsg, width_ratios=[6, 0.125], wspace=0.0
    )
    _fig_grid_gsf = _fig_grid[0, 0].subgridspec(
        nrows=3, ncols=1, wspace=0, hspace=0.125
    )

    for _ax_row, _pcm_firm2_star in enumerate((
        1.00,
        _g_bar / _divr_bar,
        _g_bar / _r_bar,
    )):
        _ax_now = _fig_2dsg.add_subplot(_fig_grid_gsf[_ax_row, 0])
        _ax_now = _set_axis_def(_ax_now, mktshares_plot_flag=True)
        _ax_now.set_xlabel(None)
        _ax_now.set_ylabel(None)
        _plt.setp(_ax_now.get_xticklabels()[1::2], visible=False)
        _plt.setp(_ax_now.get_yticklabels()[1::2], visible=False)

        _ax_now.text(
            0.81,
            0.72,
            "\n".join([
                r"$m_2 = m^* = {0:.{1}f}\%$".format(
                    (_pcmv := _pcm_firm2_star * 100), 1 * (_pcmv % 1 > 0)
                ),
                r"$m_1 \neq m_2$",
            ]),
            rotation=0,
            ha="right",
            va="top",
            fontsize=10,
            zorder=5,
        )
        if _ax_row == 0:
            # Set y-axis label
            _ax_now.yaxis.set_label_coords(-0.20, 1.0)
            _ax_now.set_ylabel(
                "Firm 2 Market Share, $s_2$",
                rotation=90,
                ha="right",
                va="top",
                fontsize=10,
            )
        elif _ax_row == 2:
            _ax_now.xaxis.set_label_coords(1.0, -0.15)
            _ax_now.set_xlabel(
                "\n".join(["Firm 1 Market Share, $s_1$"]), ha="right", fontsize=10
            )

        _qtyshr_firm1_plotter, _qtyshr_firm2_plotter, _pcm_plotter = gen_plot_data(
            market_data,
            _g_bar,
            _pcm_firm2_star,
            _inv_sel,
            h5handle=_save_data_to_file[1] if _save_data_to_file else None,
        )

        _ax_now.scatter(
            _qtyshr_firm1_plotter,
            _qtyshr_firm2_plotter,
            marker=".",
            s=(0.1 * 72.0 / _fig_2dsg.dpi) ** 2,
            c=_pcm_plotter,
            **_cmap_kwargs,
            rasterized=True,
        )
        _ax_now.set_aspect(1.0)

    gc.collect()

    # Colorbar
    _ax_cm = _fig_2dsg.add_subplot(_fig_grid[-1, -1], frameon=False)
    _ax_cm.axis("off")
    _cm_plot = _fig_2dsg.colorbar(
        cm.ScalarMappable(**_cmap_kwargs),  # type: ignore
        use_gridspec=True,
        ax=_ax_cm,
        orientation="vertical",
        fraction=3.0,
        ticks=arange(0, 1.2, 0.2),
        format=StrMethodFormatter("{x:>3.0%}"),
    )
    _cm_plot.set_label(label="Firm 1 Price-Cost Margin, $m_1$", fontsize=10)
    _cm_plot.ax.tick_params(length=5, width=0.5, labelsize=6)
    _plt.setp(
        _cm_plot.ax.yaxis.get_majorticklabels(), horizontalalignment="left", fontsize=6
    )

    _cm_plot.outline.set_visible(False)

    _base_name = DATA_DIR.joinpath(
        f"{PROG_PATH.stem}_{_hmg_pub_year}gbar{f"{_g_bar * 100:02.0f}"}PCT_{market_sample_spec.share_spec[0]}Recapture_{_inv_sel}"
    )
    _my_fig_2dsg_savepath = DATA_DIR / f"{_base_name}_2DScatterGrid.pdf"
    print(f"Save 2D plot to, {_my_fig_2dsg_savepath!r}")
    _fig_2dsg.savefig(_my_fig_2dsg_savepath, dpi=600)


if __name__ == "__main__":
    # Get Guidelines parameter values
    hmg_pub_year: Final = 2023
    inv_sel: gtl.UPPTestSpec = (
        isl.PolicySelector.ENFT,
        gtl.GUPPIWghtngSelector.MIN,
        gtl.GUPPIWghtngSelector.MIN,
    )
    r_bar, g_bar, divr_bar, *_ = getattr(
        gsf.GuidelinesStandards(hmg_pub_year),
        "safeharbor" if inv_sel[0] == isl.PolicySelector.ENFT else "presumption",
    )[2:]

    sample_sz = 10**7

    market_sample_spec = dgl.MarketSampleSpec(
        sample_sz,
        r_bar,
        share_spec=dgl.ShareSpec(
            dgl.RECConstants.INOUT,
            dgl.SHRConstants.UNI,
            DIST_PARMS_DEFAULT,
            dgl.FCOUNT_WTS_DEFAULT,
        ),
    )

    save_data_to_file_flag = False
    if save_data_to_file_flag:
        h5path = DATA_DIR / f"{PROG_PATH.stem}.h5"
        h5datafile = ptb.open_file(
            h5path,
            mode="w",
            title="Datasets, Sound GUPPI Safeharbor, Envelopes of GUPPI Boundaries",
            filters=blosc_filters,
        )
        save_data_to_file: gtl.SaveDataSpec = (
            True,
            h5datafile,
            "Intrinsic clearance stats",
        )
    else:
        save_data_to_file = False

    _main(hmg_pub_year, market_sample_spec, inv_sel, save_data_to_file)

    if save_data_to_file_flag:
        h5datafile.close()
