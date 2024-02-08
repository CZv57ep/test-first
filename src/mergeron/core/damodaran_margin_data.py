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
and valuation." Here, gross margin data compiled by Prof. Damodaran are
used to model the distribution of price-cost margin across firms that
antitrust enforcement agencies are likely to review in
merger enforcement investigations over a span of time. The
implicit assumption is that refinements in data construction from
iteration to iteration do not result in inconsistent estimates of
the empirical distribution of margins.

Second, other procedures included in this package allow the researcher to
generate margins for a single firm and impute margins of other firms in
a given industry/market based on FOCs for profit maximization by
firms facing MNL demand. In that exercise, the distribution of
inferred margins does not follow the empirical distribution estimated
from the source data, but is further constrained by the distribution of
market shares across firms.

"""

from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from types import MappingProxyType

import msgpack  # type:ignore
import numpy as np
import requests
from numpy.random import PCG64DXSM, Generator, SeedSequence
from numpy.typing import NDArray
from requests_toolbelt.downloadutils import stream  # type: ignore
from scipy import stats  # type: ignore
from xlrd import open_workbook  # type: ignore

from .. import _PKG_NAME, DATA_DIR  # noqa: TID252

__version__ = metadata.version(_PKG_NAME)


MGNDATA_ARCHIVE_PATH = DATA_DIR / "damodaran_margin_data_dict.msgpack"


def scrape_data_table(
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
    _mgn_path = _data_archive_path.parent.joinpath(f"damodaran_{_table_name}_data.xls")
    if _data_archive_path.is_file() and not data_download_flag:
        return MappingProxyType(msgpack.unpackb(_data_archive_path.read_bytes()))
    elif _mgn_path.is_file():
        _mgn_path.unlink()
        _data_archive_path.unlink()

    _REQ_TIMEOUT = (9.05, 27)
    # NYU will eventually updates its server certificate, to one signed with
    #   "InCommon RSA Server CA 2.pem", the step below will be obsolete. In
    #   the interim, it is necessary to provide the certificate chain to the
    #   root CA, so that the obsolete CA certificate is validated.
    _INCOMMON_2014_CERT_CHAIN_PATH = (
        Path(__file__).parent / "InCommon RSA Server CA cert chain.pem"
    )
    try:
        _urlopen_handle = requests.get(_mgn_urlstr, timeout=_REQ_TIMEOUT, stream=True)
    except requests.exceptions.SSLError:
        _urlopen_handle = requests.get(
            _mgn_urlstr,
            timeout=_REQ_TIMEOUT,
            stream=True,
            verify=str(_INCOMMON_2014_CERT_CHAIN_PATH),
        )

    _mgn_filename = stream.stream_response_to_file(_urlopen_handle, path=_mgn_path)

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
        _mgn_tbl_dict = scrape_data_table()

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


def resample_mgn_data(
    _sample_size: int | tuple[int, int] = (10**6, 2),
    /,
    *,
    seed_sequence: SeedSequence | None = None,
) -> NDArray[np.float64]:
    """
    Generate the specified number of draws from the empirical distribution
    for Prof. Damodaran's margin data using the estimated Gaussian KDE.
    Margins for firms in finance, investment, insurance, reinsurance, and REITs
    are excluded from the sample used to estimate the Gaussian KDE.

    Parameters
    ----------
    _sample_size
        Number of draws

    seed_sequence
        SeedSequence for seeding random-number generator when results
        are to be repeatable

    Returns
    -------
        Array of margin values

    """

    _seed_sequence = seed_sequence or SeedSequence(pool_size=8)

    _x, _w, _ = mgn_data_builder(scrape_data_table())

    _mgn_kde = stats.gaussian_kde(_x, weights=_w)

    def _generate_draws(
        _mgn_kde: stats.gaussian_kde, _ssz: int, _seed_seq: SeedSequence
    ) -> NDArray[np.float64]:
        _seed = Generator(PCG64DXSM(_seed_sequence))

        # We enlarge the sample, then truncate to
        # the range between [0.0, 1.0)
        ssz_up = int(_ssz / (_mgn_kde.integrate_box_1d(0.0, 1.0) ** 2))
        sample_1 = _mgn_kde.resample(ssz_up, seed=_seed)[0]
        return np.array(
            sample_1[(sample_1 >= 0.0) & (sample_1 <= 1)][:_ssz], np.float64
        )

    if isinstance(_sample_size, int):
        return _generate_draws(_mgn_kde, _sample_size, _seed_sequence)
    else:
        _ssz, _num_cols = _sample_size
        _ret_array = np.empty(_sample_size, np.float64)
        for _idx, _seed_seq in enumerate(_seed_sequence.spawn(_num_cols)):
            _ret_array[:, _idx] = _generate_draws(_mgn_kde, _ssz, _seed_seq)
        return _ret_array
