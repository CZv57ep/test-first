"""
Functions to parse margin data compiled by
Prof. Aswath Damodaran, Stern School of Business, NYU.

Data are downloaded or reused from a local copy, on demand.

For terms of use of Prof. Damodaran's data, please see:
https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datahistory.html

Important caveats:

Prof. Damodaran clarifies that the data construction may not be
consistent from iteration to iteration. He also notes that,
"the best use for my data is in real time corporate financial analysis
and valuation." In other words, the margin data compiled by Dr. Damodaran
may not precisely and consistently identify the distribution of margins
across firms in long-run equilibrium. Here, Prof. Damodaran's compiled
data are used to proxy the distribution of margins across firms that
antitrust enforcement agencies are likely to review in
merger enforcement investigations.

A further caveat applies here, when the researcher generates data on
a single firm's margins, with margins for all other firms in a given
industry/market being inferred based on FOCs for profit maximization
by firms facing MNL demand. In that exercise, the distribution of
inferred margins is then further constrained by the distribution of
market shares across firms.

"""

from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import msgpack  # type:ignore
import numpy as np
import seaborn as sns  # type: ignore
from numpy.random import PCG64DXSM, Generator, SeedSequence
from numpy.typing import NDArray
from scipy import stats
from xlrd import open_workbook

__version__ = metadata.version(Path(__file__).parents[1].stem)

prog_path = Path(__file__)
modl_path = prog_path.parents[1]

MGNDATA_DUMP_PATH = Path.home() / modl_path.stem / "damodaran_margin_data_dict.msgpack"


def scrape_data_table(
    _table_name: str = "margin",
    *,
    data_dump_path: Path = MGNDATA_DUMP_PATH,
    data_download_flag: bool = False,
) -> MappingProxyType[str, Mapping[str, float | int]]:
    if _table_name != "margin":  # Not validated for other tables
        raise ValueError(
            "This code is designed for parsing Prof. Damodaran's margin tables."
        )

    _mgn_urlstr = "{}{}{}".format(
        "https://pages.stern.nyu.edu/", "~adamodar/pc/datasets/", f"{_table_name}.xls"
    )
    _mgn_path = data_dump_path.parent.joinpath(f"damodaran_{_table_name}_data.xls")
    if _mgn_path.is_file() and not data_download_flag:
        return MappingProxyType(msgpack.unpackb(data_dump_path.read_bytes()))
    _, _mgn_url_hdrs = urlretrieve(url=_mgn_urlstr, filename=_mgn_path)  # noqa: S310

    _xl_book = open_workbook(_mgn_path, ragged_rows=True, on_demand=True)
    _xl_sheet = _xl_book.sheet_by_name("Industry Averages")

    _mgn_dict: dict[str, dict[str, float]] = {}
    _mgn_row_keys: list[str] = []
    _read_row_flag = False
    for _ridx in range(_xl_sheet.nrows):
        _xl_row = _xl_sheet.row_values(_ridx)
        if _xl_row[0] == "Industry Name":
            _read_row_flag = True
            _mgn_row_keys = _xl_row
            continue

        if not _xl_row[0] or not _read_row_flag:
            continue

        _xl_row[1] = int(_xl_row[1])
        _mgn_dict[_xl_row[0]] = dict(zip(_mgn_row_keys[1:], _xl_row[1:], strict=True))

    _ = data_dump_path.write_bytes(msgpack.packb(_mgn_dict))

    return MappingProxyType(_mgn_dict)


def mgn_data_builder(
    _mgn_tbl_dict: Mapping[str, Mapping[str, float | int]] | None, /
) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
    if _mgn_tbl_dict is None:
        _mgn_tbl_dict = scrape_data_table()

    _mgn_data_wts, _mgn_data_obs = np.split(
        np.array([
            tuple(_mgn_tbl_dict[_g][_h] for _h in ["Number of firms", "Gross Margin"])
            for _g in _mgn_tbl_dict
            if not _g.startswith("Total Market")
            and _g
            not in (
                "Bank (Money Center)",
                "Banks (Regional)",
                "Brokerage & Investment Banking",
                "Financial Svcs. (Non-bank & Insurance)",
                "Insurance (General)",
                "Insurance (Life)",
                "Insurance (Prop/Cas.)",
                "Investments & Asset Management",
                "R.E.I.T.",
                "Retail (REITs)",
                "Reinsurance",
            )
        ]),
        2,
        axis=1,
    )

    _mgn_wtd_avg = np.average(_mgn_data_obs, weights=_mgn_data_wts)
    # https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weighvar.pdf
    _mgn_wtd_stderr = np.sqrt(
        np.average((_mgn_data_obs - _mgn_wtd_avg) ** 2, weights=_mgn_data_wts)
        * (len(_mgn_data_wts) / (len(_mgn_data_wts) - 1))
    )

    return (
        _mgn_data_obs,
        _mgn_data_wts,
        np.round(
            (_mgn_wtd_avg, _mgn_wtd_stderr, _mgn_data_obs.min(), _mgn_data_obs.max()), 8
        ),
    )


def resample_mgn_data(
    _sample_size: int | tuple[int, ...] = (10**6, 2),
    _mgn_obs_and_wts: tuple[
        NDArray[np.floating], NDArray[np.floating] | NDArray[np.integer]
    ]
    | None = None,
    /,
    *,
    data_download_flag: bool = False,
    seed_sequence: SeedSequence | None = None,
) -> NDArray[np.float_]:
    """
    Given average margin and firm-count by industry from the margin data
    compiled by Prof. Damodaran, generate the specified number of draws
    from this empirical distribution.

    Parameters
    ----------
    _sample_size
        Number of draws

    data_download_flag
        Specify whether the latest data are to be downloaded

    seed_sequence
        SeedSequence for seeding random-number generator when results
        are to be repeatable

    Returns
    -------
        Array of margin values

    """

    if seed_sequence is None:
        seed_sequence = SeedSequence(pool_size=8)

    if _mgn_obs_and_wts is None:
        _mgn_data_obs, *_ = mgn_data_builder(scrape_data_table())
        _mgn_data_wts = np.ones_like(_mgn_data_obs)
    else:
        _mgn_data_obs, _mgn_data_wts = _mgn_obs_and_wts

    return Generator(PCG64DXSM(seed_sequence)).choice(
        _mgn_data_obs.flatten(),
        size=_sample_size,
        p=(_mgn_data_wts / _mgn_data_wts.sum()).flatten(),
    )


if __name__ == "__main__":
    mgn_data_obs, mgn_data_wts, mgn_data_stats = mgn_data_builder(
        scrape_data_table(
            "margin", data_dump_path=MGNDATA_DUMP_PATH, data_download_flag=True
        )
    )
    print(repr(mgn_data_obs))
    print(repr(mgn_data_stats))

    mgn_hist, mgn_edges = np.histogram(
        mgn_data_obs.flatten(), weights=mgn_data_wts.flatten(), bins=25, density=True
    )

    mgn_fig, mgn_ax = plt.subplots(figsize=(9, 6.5))
    plt.rcParams.update({
        "font.family": "Fira Sans",
        "font.stretch": "condensed",
        "font.weight": 300,
        "pdf.fonttype": 42,
    })

    bin_count = 25
    mgn_ax.hist(
        x=mgn_data_obs.flatten(),
        weights=mgn_data_wts.flatten(),
        bins=bin_count,
        alpha=0.6,
        density=True,
        kde=True,
        label="Downlaoded data",
        color="blue",
    )
    # Add KDE plot
    #   https://stackoverflow.com/questions/33323432
    mgn_kde = stats.gaussian_kde(x)
    mgn_xx = np.linspace(0, bin_count, 1000)
    mgn_ax.plot(mgn_xx, mgn_kde(mgn_xx), color="blue")

    sns.histplot(
        data=resample_mgn_data(10**6, (mgn_data_obs, mgn_data_wts)),
        color="green",
        alpha=0.4,
        bins=25,
        stat="density",
        ax=mgn_ax,
        label="Generated data",
    )

    mgn_ax.set_xlim(0.0, 1.0)
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

    mgn_fig.tight_layout()
    mgn_fig.savefig(MGNDATA_DUMP_PATH.parent.joinpath(f"{prog_path.stem}.pdf"))
