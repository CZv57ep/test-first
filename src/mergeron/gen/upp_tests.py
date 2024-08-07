"""
Methods to compute intrinsic clearance rates and intrinsic enforcement rates
from generated market data.

"""

from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import tables as ptb  # type: ignore
from numpy.random import SeedSequence

from .. import (  # noqa
    VERSION,
    ArrayBIGINT,
    ArrayBoolean,
    ArrayDouble,
    ArrayFloat,
    ArrayINT,
    RECForm,
    UPPAggrSelector,
)
from ..core import guidelines_boundaries as gbl  # noqa: TID252
from . import (
    EMPTY_ARRAY_DEFAULT,
    DataclassInstance,
    INVResolution,
    MarketDataSample,
    UPPTestRegime,
    UPPTestsCounts,
    UPPTestsRaw,
)
from . import enforcement_stats as esl

__version__ = VERSION

ptb.parameters.MAX_NUMEXPR_THREADS = 8
ptb.parameters.MAX_BLOSC_THREADS = 4

SaveData: TypeAlias = Literal[False] | tuple[Literal[True], ptb.File, ptb.Group]


class INVRESCntsArgs(TypedDict, total=False):
    "Keyword arguments of function, :code:`sim_enf_cnts`"

    sample_size: int
    seed_seq_list: list[SeedSequence] | None
    nthreads: int
    save_data_to_file: SaveData
    saved_array_name_suffix: str


def enf_cnts(
    _market_data_sample: MarketDataSample,
    _upp_test_parms: gbl.HMGThresholds,
    _upp_test_regime: UPPTestRegime,
    /,
) -> UPPTestsCounts:
    """Estimate enforcement and clearance counts from market data sample

    Parameters
    ----------
    _market_data_sample
        Market data sample

    _upp_test_parms
        Threshold values for various Guidelines criteria

    _upp_test_regime
        Specifies whether to analyze enforcement, clearance, or both
        and the GUPPI and diversion ratio aggregators employed, with
        default being to analyze enforcement based on the maximum
        merging-firm GUPPI and maximum diversion ratio between the
        merging firms

    Returns
    -------
    UPPTestsCounts
        Enforced and cleared counts

    """

    _enf_cnts_sim_array = -1 * np.ones((6, 2), np.int64)
    _upp_test_arrays = gen_upp_test_arrays(
        _market_data_sample, _upp_test_parms, _upp_test_regime
    )

    _fcounts, _hhi_delta, _hhi_post = (
        getattr(_market_data_sample, _g) for _g in ("fcounts", "hhi_delta", "hhi_post")
    )

    _stats_rowlen = 6
    # Clearance/enforcement counts --- by firm count
    _firmcounts_list = np.unique(_fcounts)
    if _firmcounts_list is not None and np.all(_firmcounts_list >= 0):
        _max_firmcount = max(_firmcounts_list)

        _enf_cnts_sim_byfirmcount_array = -1 * np.ones(_stats_rowlen, np.int64)
        for _firmcount in np.arange(2, _max_firmcount + 1):
            _firmcount_test = _fcounts == _firmcount

            _enf_cnts_sim_byfirmcount_array = np.vstack((
                _enf_cnts_sim_byfirmcount_array,
                np.array([
                    _firmcount,
                    np.einsum("ij->", 1 * _firmcount_test),
                    *[
                        np.einsum(
                            "ij->",
                            1 * (_firmcount_test & getattr(_upp_test_arrays, _f)),
                        )
                        for _f in _upp_test_arrays.__dataclass_fields__
                    ],
                ]),
            ))
        _enf_cnts_sim_byfirmcount_array = _enf_cnts_sim_byfirmcount_array[1:]
    else:
        _enf_cnts_sim_byfirmcount_array = np.array(
            np.nan * np.empty((1, _stats_rowlen)), np.int64
        )
        _enf_cnts_sim_byfirmcount_array[0] = 2

    # Clearance/enforcement counts --- by delta
    _hhi_delta_ranged = esl.hhi_delta_ranger(_hhi_delta)
    _enf_cnts_sim_bydelta_array = -1 * np.ones(_stats_rowlen, np.int64)
    for _hhi_delta_lim in esl.HHI_DELTA_KNOTS[:-1]:
        _hhi_delta_test = _hhi_delta_ranged == _hhi_delta_lim

        _enf_cnts_sim_bydelta_array = np.vstack((
            _enf_cnts_sim_bydelta_array,
            np.array([
                _hhi_delta_lim,
                np.einsum("ij->", 1 * _hhi_delta_test),
                *[
                    np.einsum(
                        "ij->", 1 * (_hhi_delta_test & getattr(_upp_test_arrays, _f))
                    )
                    for _f in _upp_test_arrays.__dataclass_fields__
                ],
            ]),
        ))

    _enf_cnts_sim_bydelta_array = _enf_cnts_sim_bydelta_array[1:]

    # Clearance/enforcement counts --- by zone
    try:
        _hhi_zone_post_ranged = esl.hhi_zone_post_ranger(_hhi_post)
    except ValueError as _err:
        print(_hhi_post)
        raise _err

    _stats_byconczone_sim = -1 * np.ones(_stats_rowlen + 1, np.int64)
    for _hhi_zone_post_knot in esl.HHI_POST_ZONE_KNOTS[:-1]:
        _level_test = _hhi_zone_post_ranged == _hhi_zone_post_knot

        for _hhi_zone_delta_knot in [0, 100, 200]:
            _delta_test = (
                _hhi_delta_ranged > 100
                if _hhi_zone_delta_knot == 200
                else _hhi_delta_ranged == _hhi_zone_delta_knot
            )

            _conc_test = _level_test & _delta_test

            _stats_byconczone_sim = np.vstack((
                _stats_byconczone_sim,
                np.array([
                    _hhi_zone_post_knot,
                    _hhi_zone_delta_knot,
                    np.einsum("ij->", 1 * _conc_test),
                    *[
                        np.einsum(
                            "ij->", 1 * (_conc_test & getattr(_upp_test_arrays, _f))
                        )
                        for _f in _upp_test_arrays.__dataclass_fields__
                    ],
                ]),
            ))

    _enf_cnts_sim_byconczone_array = esl.enf_cnts_byconczone(_stats_byconczone_sim[1:])
    del _stats_byconczone_sim
    del _hhi_delta, _hhi_post, _fcounts

    return UPPTestsCounts(
        _enf_cnts_sim_byfirmcount_array,
        _enf_cnts_sim_bydelta_array,
        _enf_cnts_sim_byconczone_array,
    )


def gen_upp_test_arrays(
    _market_data: MarketDataSample,
    _upp_test_parms: gbl.HMGThresholds,
    _sim_test_regime: UPPTestRegime,
    /,
) -> UPPTestsRaw:
    """
    Generate UPP tests arrays for given configuration and market sample

    Given a standards vector, market

    Parameters
    ----------
    _market_data
        market data sample
    _upp_test_parms
        guidelines thresholds for testing UPP and related statistics
    _sim_test_regime
        configuration to use for generating UPP tests

    """
    _g_bar, _divr_bar, _cmcr_bar, _ipr_bar = (
        getattr(_upp_test_parms, _f) for _f in ("guppi", "divr", "cmcr", "ipr")
    )

    _guppi_array, _ipr_array, _cmcr_array = (
        np.empty_like(_market_data.price_array) for _ in range(3)
    )

    np.einsum(
        "ij,ij,ij->ij",
        _market_data.divr_array,
        _market_data.pcm_array[:, ::-1],
        _market_data.price_array[:, ::-1] / _market_data.price_array,
        out=_guppi_array,
    )

    np.divide(
        np.einsum("ij,ij->ij", _market_data.pcm_array, _market_data.divr_array),
        1 - _market_data.divr_array,
        out=_ipr_array,
    )

    np.divide(_ipr_array, 1 - _market_data.pcm_array, out=_cmcr_array)

    (_divr_test_vector,) = _compute_test_value_seq(
        (_market_data.divr_array,),
        _market_data.frmshr_array,
        _sim_test_regime.divr_aggregator,
    )

    (_guppi_test_vector, _cmcr_test_vector, _ipr_test_vector) = _compute_test_value_seq(
        (_guppi_array, _cmcr_array, _ipr_array),
        _market_data.frmshr_array,
        _sim_test_regime.guppi_aggregator,
    )
    del _cmcr_array, _ipr_array, _guppi_array

    if _sim_test_regime.resolution == INVResolution.ENFT:
        _upp_test_arrays = UPPTestsRaw(
            _guppi_test_vector >= _g_bar,
            (_guppi_test_vector >= _g_bar) | (_divr_test_vector >= _divr_bar),
            _cmcr_test_vector >= _cmcr_bar,
            _ipr_test_vector >= _ipr_bar,
        )
    else:
        _upp_test_arrays = UPPTestsRaw(
            _guppi_test_vector < _g_bar,
            (_guppi_test_vector < _g_bar) & (_divr_test_vector < _divr_bar),
            _cmcr_test_vector < _cmcr_bar,
            _ipr_test_vector < _ipr_bar,
        )

    return _upp_test_arrays


def _compute_test_value_seq(
    _test_measure_seq: tuple[ArrayDouble, ...],
    _wt_array: ArrayDouble,
    _aggregator: UPPAggrSelector,
) -> tuple[ArrayDouble, ...]:
    _wt_array = (
        _wt_array / np.einsum("ij->i", _wt_array)[:, None]
        if _aggregator
        in (
            UPPAggrSelector.CPA,
            UPPAggrSelector.CPD,
            UPPAggrSelector.OSA,
            UPPAggrSelector.OSD,
        )
        else EMPTY_ARRAY_DEFAULT
    )

    match _aggregator:
        case UPPAggrSelector.AVG:
            _test_value_seq = (
                1 / 2 * np.einsum("ij->i", _g)[:, None] for _g in _test_measure_seq
            )
        case UPPAggrSelector.CPA:
            _test_value_seq = (
                np.einsum("ij,ij->i", _wt_array[:, ::-1], _g)[:, None]
                for _g in _test_measure_seq
            )
        case UPPAggrSelector.CPD:
            _test_value_seq = (
                np.sqrt(np.einsum("ij,ij,ij->i", _wt_array[:, ::-1], _g, _g))[:, None]
                for _g in _test_measure_seq
            )
        case UPPAggrSelector.DIS:
            _test_value_seq = (
                np.sqrt(1 / 2 * np.einsum("ij,ij->i", _g, _g))[:, None]
                for _g in _test_measure_seq
            )
        case UPPAggrSelector.MAX:
            _test_value_seq = (
                _g.max(axis=1, keepdims=True) for _g in _test_measure_seq
            )
        case UPPAggrSelector.MIN:
            _test_value_seq = (
                _g.min(axis=1, keepdims=True) for _g in _test_measure_seq
            )
        case UPPAggrSelector.OSA:
            _test_value_seq = (
                np.einsum("ij,ij->i", _wt_array, _g)[:, None]
                for _g in _test_measure_seq
            )
        case UPPAggrSelector.OSD:
            _test_value_seq = (
                np.sqrt(np.einsum("ij,ij,ij->i", _wt_array, _g, _g))[:, None]
                for _g in _test_measure_seq
            )
        case _:
            raise ValueError("GUPPI/diversion ratio aggregation method is invalid.")
    return tuple(_test_value_seq)


def initialize_hd5(
    _h5_path: Path, _hmg_pub_year: gbl.HMGPubYear, _test_regime: UPPTestRegime, /
) -> tuple[SaveData, str]:
    _h5_title = f"HMG version: {_hmg_pub_year}; Test regime: {_test_regime}"
    if _h5_path.is_file():
        _h5_path.unlink()
    _h5_file = ptb.open_file(_h5_path, mode="w", title=_h5_title)
    _save_data_to_file: tuple[Literal[True], ptb.File, str] = (True, _h5_file, "/")
    _next_subgroup_name_root = "enf_{}_{}_{}_{}".format(
        _hmg_pub_year,
        *(getattr(_test_regime, _f.name).name for _f in _test_regime.__attrs_attrs__),
    )
    return _save_data_to_file, _next_subgroup_name_root


def save_data_to_hdf5(
    _dclass: DataclassInstance,
    /,
    *,
    saved_array_name_suffix: str | None = "",
    excluded_attrs: Sequence[str] | None = (),
    save_data_to_file: SaveData = False,
) -> None:
    if save_data_to_file:
        _, _h5_file, _h5_group = save_data_to_file
        # Save market data arrays
        excluded_attrs = excluded_attrs or ()
        for _array_name in _dclass.__dataclass_fields__:
            if _array_name in excluded_attrs:
                continue
            save_array_to_hdf5(
                getattr(_dclass, _array_name),
                _array_name,
                _h5_group,
                _h5_file,
                saved_array_name_suffix=saved_array_name_suffix,
            )


def save_array_to_hdf5(
    _array_obj: ArrayFloat | ArrayINT | ArrayDouble | ArrayBIGINT | ArrayBoolean,
    _array_name: str,
    _h5_group: ptb.Group,
    _h5_file: ptb.File,
    /,
    *,
    saved_array_name_suffix: str | None = None,
) -> None:
    _h5_array_name = f"{_array_name}_{saved_array_name_suffix or ""}".rstrip("_")

    with suppress(ptb.NoSuchNodeError):
        _h5_file.remove_node(_h5_group, name=_array_name)

    _h5_array = ptb.CArray(
        _h5_group,
        _h5_array_name,
        atom=ptb.Atom.from_dtype(_array_obj.dtype),
        shape=_array_obj.shape,
        filters=ptb.Filters(complevel=3, complib="blosc:lz4hc", fletcher32=True),
    )
    _h5_array[:] = _array_obj


if __name__ == "__main__":
    print(
        "This module defines classes with methods for generating UPP test arrays and UPP test-counts arrays on given data."
    )
