"""
Functions to parse margin data compiled by
Prof. Aswath Damodaran, Stern School of Business, NYU.

Data are downloaded or reused from a local copy, on demand.

For terms of use of Prof. Damodaran's data, please see:
https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datahistory.html

NOTES
-----

Prof. Damodaran notes that the data construction may not be
consistent from iteration to iteration. He also notes that,
"the best use for my data is in real time corporate financial analysis
and valuation." Here, gross margin data compiled by Prof. Damodaran are
optionally used to model the distribution of price-cost margin
across firms that antitrust enforcement agencies are likely to review in
merger enforcement investigations over a multi-year span. The
implicit assumption is that refinements in source-data construction from
iteration to iteration do not result in inconsistent estimates of
the empirical distribution of margins estimated using
a Gaussian kernel density estimator (KDE).

Second, other procedures included in this package allow the researcher to
generate margins for a single firm and impute margins of other firms in
a model relevant antitrust market based on FOCs for profit maximization by
firms facing MNL demand. In that exercise, the distribution of
inferred margins does not follow the empirical distribution estimated
from the source data, due to restrictions resulting from the distribution of
generated market shares across firms and the feasibility condition that
price-cost margins fall in the interval :math:`[0, 1]`.

"""

import shutil
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from types import MappingProxyType

import msgpack  # type:ignore
import numpy as np
import urllib3
from numpy.random import PCG64DXSM, Generator, SeedSequence
from numpy.typing import NDArray
from scipy import stats  # type: ignore
from xlrd import open_workbook  # type: ignore

from .. import _PKG_NAME, DATA_DIR, VERSION  # noqa: TID252

__version__ = VERSION

MGNDATA_ARCHIVE_PATH = DATA_DIR / "damodaran_margin_data_dict.msgpack"

u3pm = urllib3.PoolManager()


def mgn_data_getter(  # noqa: PLR0912
    _table_name: str = "margin",
    *,
    data_archive_path: Path | None = None,
    data_download_flag: bool = False,
) -> MappingProxyType[str, Mapping[str, float | int]]:
    if _table_name != "margin":  # Not validated for other tables
        raise ValueError(
            "This code is designed for parsing Prof. Damodaran's margin tables."
        )

    _data_archive_path = data_archive_path or MGNDATA_ARCHIVE_PATH

    _mgn_urlstr = f"https://pages.stern.nyu.edu/~adamodar/pc/datasets/{_table_name}.xls"
    _mgn_path = _data_archive_path.parent / f"damodaran_{_table_name}_data.xls"
    if _data_archive_path.is_file() and not data_download_flag:
        return MappingProxyType(msgpack.unpackb(_data_archive_path.read_bytes()))
    elif _mgn_path.is_file():
        _mgn_path.unlink()
        if _data_archive_path.is_file():
            _data_archive_path.unlink()

    try:
        _chunk_size = 1024 * 1024
        with (
            u3pm.request("GET", _mgn_urlstr, preload_content=False) as _urlopen_handle,
            _mgn_path.open("wb") as _mgn_file,
        ):
            while True:
                _data = _urlopen_handle.read(_chunk_size)
                if not _data:
                    break
                _mgn_file.write(_data)

        print(f"Downloaded {_mgn_urlstr} to {_mgn_path}.")

    except urllib3.exceptions.MaxRetryError as _err:
        if isinstance(_err.__cause__, urllib3.exceptions.SSLError):
            # Works fine with other sites secured with certificates
            # from the Internet2 CA, such as,
            # https://snap.stanford.edu/data/web-Stanford.txt.gz
            print(
                f"WARNING: Could not establish secure connection to, {_mgn_urlstr}."
                "Using bundled copy."
            )
            if not _mgn_path.is_file():
                with resources.as_file(
                    resources.files(f"{_PKG_NAME}.data").joinpath(
                        "damodaran_margin_data.xls"
                    )
                ) as _mgn_data_archive_path:
                    shutil.copy2(_mgn_data_archive_path, _mgn_path)
        else:
            raise _err

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

    _ = _data_archive_path.write_bytes(msgpack.packb(_mgn_dict))

    return MappingProxyType(_mgn_dict)


def mgn_data_builder(
    _mgn_tbl_dict: Mapping[str, Mapping[str, float | int]] | None = None, /
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if _mgn_tbl_dict is None:
        _mgn_tbl_dict = mgn_data_getter()

    _mgn_data_wts, _mgn_data_obs = (
        _f.flatten()
        for _f in np.hsplit(
            np.array([
                tuple(
                    _mgn_tbl_dict[_g][_h] for _h in ["Number of firms", "Gross Margin"]
                )
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
        )
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


def mgn_data_resampler(
    _sample_size: int | tuple[int, int] = (10**6, 2),
    /,
    *,
    seed_sequence: SeedSequence | None = None,
) -> NDArray[np.float64]:
    """
    Generate draws from the empirical distribution bassed on Prof. Damodaran's margin data.

    The empirical distribution is estimated using a Gaussian KDE; the bandwidth
    selected using Silverman's rule is narrowed to reflect that the margin data
    are multimodal. Margins for firms in finance, investment, insurance, reinsurance, and
    REITs are excluded from the sample used to estimate the empirical distribution.

    Parameters
    ----------
    _sample_size
        Number of draws; if tuple, (number of draws, number of columns)

    seed_sequence
        SeedSequence for seeding random-number generator when results
        are to be repeatable

    Returns
    -------
        Array of margin values

    """

    _seed_sequence = seed_sequence or SeedSequence(pool_size=8)

    _x, _w, _ = mgn_data_builder(mgn_data_getter())

    _mgn_kde = stats.gaussian_kde(_x, weights=_w, bw_method="silverman")
    _mgn_kde.set_bandwidth(bw_method=_mgn_kde.factor / 3.0)

    if isinstance(_sample_size, int):
        return np.array(
            _mgn_kde.resample(_sample_size, seed=Generator(PCG64DXSM(_seed_sequence)))[
                0
            ]
        )
    elif isinstance(_sample_size, tuple) and len(_sample_size) == 2:
        _ssz, _num_cols = _sample_size
        _ret_array = np.empty(_sample_size, np.float64)
        for _idx, _seed_seq in enumerate(_seed_sequence.spawn(_num_cols)):
            _ret_array[:, _idx] = _mgn_kde.resample(
                _ssz, seed=Generator(PCG64DXSM(_seed_seq))
            )[0]
        return _ret_array
    else:
        raise ValueError(f"Invalid sample size: {_sample_size!r}")
