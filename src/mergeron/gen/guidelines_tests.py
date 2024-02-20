"""
Routines to estimateintrinsic clearnace rates and intrinsic enforcement rates
from generated market data.

"""

from importlib.metadata import version

from .. import _PKG_NAME  # noqa: TID252

__version__ = version(_PKG_NAME)


import enum
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeAlias

import numpy as np
import tables as ptb  # type: ignore
from attr import evolve
from joblib import Parallel, cpu_count, delayed  # type: ignore
from numpy.random import SeedSequence
from numpy.typing import NDArray

from ..core import guidelines_standards as gsf  # noqa: TID252
from . import data_generation as dgl
from . import investigations_stats as isl

ptb.parameters.MAX_NUMEXPR_THREADS = 8
ptb.parameters.MAX_BLOSC_THREADS = 4

SaveData: TypeAlias = Literal[False] | tuple[Literal[True], ptb.File, str]


@enum.unique
class GUPPIWghtngSelector(enum.StrEnum):
    """
    Aggregator selection for GUPPI and diversion ratio

    """

    AVG = "average"
    CPA = "cross-product-share-weighted average"
    CPD = "cross-product-share-weighted distance"
    DIS = "symmetrically-weighted distance"
    MAX = "max"
    MIN = "min"
    OSA = "own-share-weighted average"
    OSD = "own-share-weighted distance"


UPPTestRegime: TypeAlias = tuple[
    isl.PolicySelector, GUPPIWghtngSelector, GUPPIWghtngSelector | None
]


class UPPTests(NamedTuple):
    guppi_test_simple: NDArray[np.bool_]
    guppi_test_compound: NDArray[np.bool_]
    cmcr_test: NDArray[np.bool_]
    ipr_test: NDArray[np.bool_]


def sim_invres_cnts_ll(
    _invres_parm_vec: gsf.GuidelinesSTD,
    _mkt_sample_spec: dgl.MarketSampleSpec,
    _sim_invres_cnts_kwargs: Mapping[str, Any],
    /,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """
    A function to parallelize simulations

    The parameters _sim_invres_cnts_kwargs is passed unaltered to
    the parent function, sim_invres_cnts(), except that, if provided,
    "seed_seq_list" is used to spawn a seed sequence for each thread,
    to assure independent samples in each thread. The number of draws
    in each thread may be tuned, by trial and error, to the amount of
    memory (RAM) available.

    """

    _sample_sz = _mkt_sample_spec.sample_size
    _subsample_sz = 10**6
    _iter_count = int(_sample_sz / _subsample_sz) if _subsample_sz < _sample_sz else 1
    _thread_count = cpu_count()

    # Crate a copy, to avoid side effects in the outer scope
    _mkt_sample_spec_here = evolve(_mkt_sample_spec, sample_size=_subsample_sz)

    _rng_seed_seq_list = [None] * _iter_count
    if _sim_invres_cnts_kwargs:
        if _sseql := _sim_invres_cnts_kwargs.get("seed_seq_list", None):
            _rng_seed_seq_list = list(
                zip(*[g.spawn(_iter_count) for g in _sseql], strict=True)  # type: ignore
            )

        _sim_invres_cnts_ll_kwargs = {
            _k: _v
            for _k, _v in _sim_invres_cnts_kwargs.items()
            if _k != "seed_seq_list"
        }
    else:
        _sim_invres_cnts_ll_kwargs = {}

    _res_list = Parallel(n_jobs=_thread_count, prefer="threads")(
        delayed(sim_invres_cnts)(
            _invres_parm_vec,
            _mkt_sample_spec_here,
            **_sim_invres_cnts_ll_kwargs,
            saved_array_name_suffix=f"{_thread_id:0{int(np.ceil(np.log10(_thread_count)))}d}",
            seed_seq_list=_rng_seed_seq_list_ch,
        )
        for _thread_id, _rng_seed_seq_list_ch in enumerate(_rng_seed_seq_list)
    )

    _res_list_stacks = [np.stack([_j[_k] for _j in _res_list]) for _k in range(3)]
    (
        _invres_cnts_sim_byfirmcount_array,
        _invres_cnts_sim_bydelta_array,
        _invres_cnts_sim_byconczone_array,
    ) = (
        np.column_stack((
            _g[0, :, :_h],
            np.einsum("ijk->jk", np.int64(1) * _g[:, :, _h:]),
        ))
        for _g, _h in zip(_res_list_stacks, [1, 1, 3], strict=True)
    )
    del _res_list, _res_list_stacks

    return (
        _invres_cnts_sim_byfirmcount_array,
        _invres_cnts_sim_bydelta_array,
        _invres_cnts_sim_byconczone_array,
    )


def sim_invres_cnts(
    _guppi_test_parms: gsf.GuidelinesSTD,
    _mkt_sample_spec: dgl.MarketSampleSpec,
    /,
    *,
    sim_test_regime: tuple[
        isl.PolicySelector, GUPPIWghtngSelector, GUPPIWghtngSelector | None
    ],
    saved_array_name_suffix: str = "",
    save_data_to_file: SaveData = False,
    seed_seq_list: list[SeedSequence] | None = None,
    nthreads: int = 16,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    if _mkt_sample_spec.recapture_rate != _guppi_test_parms.rec:
        raise ValueError(
            "{} {} {} {}".format(
                f"Value, {_mkt_sample_spec.recapture_rate}",
                "of recapture rate in the second positional argument",
                f"must equal its value, {_guppi_test_parms.rec}",
                "in the first positional argument.",
            )
        )

    # Generate market data
    _market_data = dgl.gen_market_sample(
        _mkt_sample_spec, seed_seq_list=seed_seq_list, nthreads=nthreads
    )

    _invalid_array_names = (
        ("fcounts", "choice_prob_outgd", "nth_firm_share", "hhi_post")
        if _mkt_sample_spec.share_spec.dist_type == "Uniform"
        else ()
    )

    save_namedtuple_to_hdf5(
        _market_data,
        saved_array_name_suffix,
        _invalid_array_names,
        save_data_to_file=save_data_to_file,
    )

    _upp_tests_data = gen_upp_arrays(
        _guppi_test_parms,
        _market_data,
        sim_test_regime,
        saved_array_name_suffix=saved_array_name_suffix,
        save_data_to_file=save_data_to_file,
    )

    _fcounts, _hhi_delta, _hhi_post = (
        getattr(_market_data, _g) for _g in ["fcounts", "hhi_delta", "hhi_post"]
    )
    del _market_data

    # Clearance/enforcement counts --- by firm count
    # Accumulate firm_count, numobs, num_gmbound, num_gsf, num_cbound, num_ibound
    _stats_rowlen = 6
    _firm_counts_prob_weights: NDArray[np.float64 | np.int64] = (
        dgl.FCOUNT_WTS_DEFAULT
        if _mkt_sample_spec.share_spec.firm_counts_prob_weights is None
        else _mkt_sample_spec.share_spec.firm_counts_prob_weights
    )
    _max_firm_count = len(_firm_counts_prob_weights)

    _invres_cnts_sim_byfirmcount_array = -1 * np.ones(_stats_rowlen, np.int64)
    for _firm_cnt in 2 + np.arange(_max_firm_count):
        _firm_count_test = _fcounts == _firm_cnt

        _invres_cnts_sim_byfirmcount_array = np.row_stack((
            _invres_cnts_sim_byfirmcount_array,
            np.array([
                _firm_cnt,
                np.einsum("ij->", 1 * _firm_count_test),
                *[
                    np.einsum("ij->", 1 * (_firm_count_test & _f))
                    for _f in _upp_tests_data
                ],
            ]),
        ))
    _invres_cnts_sim_byfirmcount_array = _invres_cnts_sim_byfirmcount_array[1:]

    _hhi_delta_ranged = isl.hhi_delta_ranger(_hhi_delta)
    _invres_cnts_sim_bydelta_array = -1 * np.ones(_stats_rowlen, np.int64)
    for _hhi_delta_lim in isl.HHI_DELTA_KNOTS[:-1]:
        _hhi_delta_test = _hhi_delta_ranged == _hhi_delta_lim

        _invres_cnts_sim_bydelta_array = np.row_stack((
            _invres_cnts_sim_bydelta_array,
            np.array([
                _hhi_delta_lim,
                np.einsum("ij->", 1 * _hhi_delta_test),
                *[
                    np.einsum("ij->", 1 * (_hhi_delta_test & _f))
                    for _f in _upp_tests_data
                ],
            ]),
        ))

    _invres_cnts_sim_bydelta_array = _invres_cnts_sim_bydelta_array[1:]

    # Clearance/enfrocement counts --- by zone, coded
    try:
        _hhi_zone_post_ranged = isl.hhi_zone_post_ranger(_hhi_post)
    except ValueError as _err:
        print(_hhi_post)
        raise _err

    _stats_byconczone_sim = -1 * np.ones(_stats_rowlen + 1, np.int64)
    for _hhi_zone_post_knot in isl.HHI_POST_ZONE_KNOTS[:-1]:
        _level_test = _hhi_zone_post_ranged == _hhi_zone_post_knot

        for _hhi_zone_delta_knot in [0, 100, 200]:
            _delta_test = (
                _hhi_delta_ranged > 100
                if _hhi_zone_delta_knot == 200
                else _hhi_delta_ranged == _hhi_zone_delta_knot
            )

            _conc_test = _level_test & _delta_test

            _stats_byconczone_sim = np.row_stack((
                _stats_byconczone_sim,
                np.array([
                    _hhi_zone_post_knot,
                    _hhi_zone_delta_knot,
                    np.einsum("ij->", 1 * _conc_test),
                    *[
                        np.einsum("ij->", 1 * (_conc_test & _f))
                        for _f in _upp_tests_data
                    ],
                ]),
            ))

    _invres_cnts_sim_byconczone_array = isl.invres_cnts_byconczone(
        _stats_byconczone_sim[1:]
    )
    del _stats_byconczone_sim
    del _hhi_delta, _hhi_post, _fcounts

    return (
        _invres_cnts_sim_byfirmcount_array,
        _invres_cnts_sim_bydelta_array,
        _invres_cnts_sim_byconczone_array,
    )


def gen_upp_arrays(
    _guppi_test_parms: gsf.GuidelinesSTD,
    _market_data: dgl.MarketsSample,
    _sim_test_regime: tuple[
        isl.PolicySelector, GUPPIWghtngSelector, GUPPIWghtngSelector | None
    ],
    /,
    *,
    saved_array_name_suffix: str = "",
    save_data_to_file: SaveData = False,
) -> UPPTests:
    _g_bar, _divr_bar, _cmcr_bar, _ipr_bar = (
        getattr(_guppi_test_parms, _f) for _f in ("guppi", "divr", "cmcr", "ipr")
    )

    _invres_select, _guppi_wgtng_policy, _divr_wgtng_policy = _sim_test_regime

    _guppi_array = np.empty_like(_market_data.divr_array)
    np.einsum(
        "ij,ij,ij->ij",
        _market_data.divr_array,
        _market_data.pcm_array[:, ::-1],
        _market_data.price_array[:, ::-1] / _market_data.price_array,
        out=_guppi_array,
    )

    _cmcr_array = np.empty_like(_market_data.divr_array)
    np.divide(
        np.einsum("ij,ij->ij", _market_data.pcm_array, _market_data.divr_array),
        np.einsum("ij,ij->ij", 1 - _market_data.pcm_array, 1 - _market_data.divr_array),
        out=_cmcr_array,
    )

    _ipr_array = np.empty_like(_market_data.divr_array)
    np.divide(
        np.einsum("ij,ij->ij", _market_data.pcm_array, _market_data.divr_array),
        1 - _market_data.divr_array,
        out=_ipr_array,
    )

    # This one needs further testing:
    # _ipr_array_alt = np.empty_like(_market_data.divr_array)
    # np.divide(_guppi_array, (1 - _market_data.divr_array[:, ::-1]), out=_ipr_array_alt)

    _test_measure_seq = (_market_data.divr_array, _guppi_array, _cmcr_array, _ipr_array)

    _wt_array = (
        _market_data.frmshr_array
        / np.einsum("ij->i", _market_data.frmshr_array)[:, None]
        if _guppi_wgtng_policy
        in (
            GUPPIWghtngSelector.CPA,
            GUPPIWghtngSelector.CPD,
            GUPPIWghtngSelector.OSA,
            GUPPIWghtngSelector.OSD,
        )
        else dgl.EMPTY_ARRAY_DEFAULT
    )

    match _guppi_wgtng_policy:
        case GUPPIWghtngSelector.AVG:
            _test_value_seq = (
                1 / 2 * np.einsum("ij->i", _g)[:, None] for _g in _test_measure_seq
            )
        case GUPPIWghtngSelector.CPA:
            _test_value_seq = (
                np.einsum("ij,ij->i", _wt_array[:, ::-1], _g)[:, None]
                for _g in _test_measure_seq
            )
        case GUPPIWghtngSelector.CPD:
            _test_value_seq = (
                np.sqrt(np.einsum("ij,ij,ij->i", _wt_array[:, ::-1], _g, _g))[:, None]
                for _g in _test_measure_seq
            )
        case GUPPIWghtngSelector.DIS:
            _test_value_seq = (
                np.sqrt(1 / 2 * np.einsum("ij,ij->i", _g, _g))[:, None]
                for _g in _test_measure_seq
            )
        case GUPPIWghtngSelector.MAX:
            _test_value_seq = (
                _g.max(axis=1, keepdims=True) for _g in _test_measure_seq
            )
        case GUPPIWghtngSelector.MIN:
            _test_value_seq = (
                _g.min(axis=1, keepdims=True) for _g in _test_measure_seq
            )
        case GUPPIWghtngSelector.OSA:
            _test_value_seq = (
                np.einsum("ij,ij->i", _wt_array, _g)[:, None]
                for _g in _test_measure_seq
            )
        case GUPPIWghtngSelector.OSD:
            _test_value_seq = (
                np.sqrt(np.einsum("ij,ij,ij->i", _wt_array, _g, _g))[:, None]
                for _g in _test_measure_seq
            )
        case _:
            raise ValueError("GUPPI/diversion ratio aggregation method is invalid.")
    del _cmcr_array, _guppi_array
    (_divr_test_vector, _guppi_test_vector, _cmcr_test_vector, _ipr_test_vector) = (
        _test_value_seq
    )

    if _divr_wgtng_policy == GUPPIWghtngSelector.MAX:
        _divr_test_vector = _market_data.divr_array.max(axis=1, keepdims=True)

    if _invres_select == isl.PolicySelector.ENFT:
        _upp_tests_data = UPPTests(
            _guppi_test_vector >= _g_bar,
            (_guppi_test_vector >= _g_bar) | (_divr_test_vector >= _divr_bar),
            _cmcr_test_vector >= _cmcr_bar,
            _ipr_test_vector >= _ipr_bar,
        )
    else:
        _upp_tests_data = UPPTests(
            _guppi_test_vector < _g_bar,
            (_guppi_test_vector < _g_bar) & (_divr_test_vector < _divr_bar),
            _cmcr_test_vector < _cmcr_bar,
            _ipr_test_vector < _ipr_bar,
        )
    del _guppi_test_vector, _divr_test_vector, _cmcr_test_vector, _ipr_test_vector

    save_namedtuple_to_hdf5(
        _upp_tests_data,
        saved_array_name_suffix,
        (),
        save_data_to_file=save_data_to_file,
    )

    return _upp_tests_data


def save_namedtuple_to_hdf5(
    _dclass: NamedTuple,
    _saved_array_name_suffix: str,
    _excl_attrs: Sequence[str] = (),
    /,
    *,
    save_data_to_file: SaveData = False,
) -> None:
    if save_data_to_file:
        _, _h5_datafile, _h5_hier = save_data_to_file
        # Save market data arrays
        for _array_name in _dclass._fields:
            if _excl_attrs and _array_name in _excl_attrs:
                pass
            save_to_hdf(
                _dclass,
                _array_name,
                _h5_datafile,
                _h5_hier,
                saved_array_name_suffix=_saved_array_name_suffix,
            )


def save_to_hdf(
    _dclass: NamedTuple,
    _array_name: str,
    _h5_datafile: ptb.File,
    _h5_hier: str,
    /,
    *,
    saved_array_name_suffix: str | None = None,
) -> None:
    _data_array = getattr(_dclass, _array_name)
    _h5_array_name = (
        f"{_array_name}_{saved_array_name_suffix}"
        if saved_array_name_suffix
        else _array_name
    )
    print(
        f"PyTables: Now saving array, {_h5_array_name!r}, at node, {f'"{_h5_hier}"'}."
    )
    _h5_array = _h5_datafile.create_carray(
        _h5_hier,
        _h5_array_name,
        obj=_data_array,
        createparents=True,
        title=f"{_array_name})",
    )
    _h5_array[:] = _data_array
