"""
Plot the empirical distribution of margin data downloaded from
Prof. Damodaran's website at NYU.

"""

import warnings
from pathlib import Path

import numpy as np
from matplotlib.ticker import StrMethodFormatter
from numpy.typing import NDArray
from scipy import stats  # type: ignore

import mergeron.core.damodaran_margin_data as dmgn
from mergeron import DATA_DIR
from mergeron.core.guidelines_boundaries import boundary_plot

PROG_PATH = Path(__file__)


def _get_margin_data() -> (
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
):
    return dmgn.mgn_data_builder()


if __name__ == "__main__":
    mgn_data_obs, mgn_data_wts, mgn_data_stats = _get_margin_data()
    print(repr(mgn_data_obs))
    print(repr(mgn_data_stats))

    plt, mgn_fig, mgn_ax, set_axis_def = boundary_plot(mktshares_plot_flag=False)
    mgn_fig.set_figheight(6.5)
    mgn_fig.set_figwidth(9.0)

    bin_count = 25
    _, mgn_bins, _ = mgn_ax.hist(
        x=mgn_data_obs,
        weights=mgn_data_wts,
        bins=bin_count,
        alpha=0.4,
        density=True,
        label="Downloaded data",
        color="#004488",  # Paul Tol's High Contrast Blue
    )
    mgn_ax_yticklabels = mgn_ax.get_yticklabels()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # Don't warn regarding the below; ticklabels have been fixed before this point
        mgn_ax.set_yticklabels([
            f"{float(_g.get_text()) * np.diff(mgn_bins)[-1]:.0%}"
            for _g in mgn_ax_yticklabels
        ])

    # Add KDE plot
    #   https://stackoverflow.com/questions/33323432
    mgn_kde = stats.gaussian_kde(mgn_data_obs, weights=mgn_data_wts)
    print(
        f"Approximately {mgn_kde.integrate_box_1d(0, 1):.2%} of the estimated mass",
        "of the empirical distribution under the Gaussian KDE",
        "estimated using selected margin data falls in the interval [0, 1].",
    )
    mgn_xx = np.linspace(0, bin_count, 10**5)
    mgn_ax.plot(
        mgn_xx,
        mgn_kde(mgn_xx),
        color="#004488",
        rasterized=True,
        label="Estimated Density",
    )

    sample_size = 10**6
    mgn_ax.hist(
        x=dmgn.resample_mgn_data(sample_size),
        color="#DDAA33",  # Paul Tol's High Contrast Yellow
        alpha=0.6,
        bins=bin_count,
        density=True,
        label="Generated data",
    )

    mgn_ax.legend(
        loc="best",
        fancybox=False,
        shadow=False,
        frameon=True,
        facecolor="white",
        edgecolor="white",
        framealpha=1,
        fontsize="small",
    )

    mgn_ax.set_xlim(0.0, 1.0)
    mgn_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:>3.0%}"))
    mgn_ax.set_xlabel("Price Cost Margin", fontsize=10)
    mgn_ax.set_ylabel("Relative Frequency", fontsize=10)

    mgn_fig.tight_layout()
    plt.savefig(PROG_PATH.with_suffix(".pdf"))
