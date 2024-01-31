"""
Functions to generate a sample of mergers and to estimate
intrinsic clearnace rates and intrinsic enforcement rates.

"""

import enum
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import tables as ptb  # type: ignore
from attr import evolve
from joblib import Parallel, cpu_count, delayed  # type: ignore
from numpy.random import SeedSequence
from numpy.typing import NDArray

import mergeron.gen.data_generation as dgl
import mergeron.gen.investigations_stats as clstl

__version__ = metadata.version(Path(__file__).parents[1].stem)

mod_path = Path(__file__)
data_path = Path.home() / mod_path.parent.stem

ptb.parameters.MAX_NUMEXPR_THREADS = 8
ptb.parameters.MAX_BLOSC_THREADS = 4


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


class UPPTests(NamedTuple):
    guppi_test_simple: NDArray[np.bool_]
    guppi_test_compound: NDArray[np.bool_]
    cmcr_test: NDArray[np.bool_]
    ipr_test: NDArray[np.bool_]


def sim_clrenf_cnts_ll(
    _clrenf_parm_vec: Sequence[float],
    _mkt_sample_spec: dgl.MarketSampleSpec,
    _sim_clrenf_cnts_kwargs: Mapping[str, Any],
    /,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """
    A function to parallelize simulations

    The parameters _sim_clrenf_cnts_kwargs is passed unaltered to
    the parent function, sim_clrenf_cnts(), except that, if provided,
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

    if _sim_clrenf_cnts_kwargs:
        if (_sseql := _sim_clrenf_cnts_kwargs.get("seed_seq_list", None)) is None:
            _rng_seed_seq_list = [None] * _iter_count
        else:
            _rng_seed_seq_list = list(
                zip(*[g.spawn(_iter_count) for g in _sseql], strict=True)  # type: ignore
            )

        _sim_clrenf_cnts_ll_kwargs = {
            _k: _sim_clrenf_cnts_kwargs[_k]
            for _k in _sim_clrenf_cnts_kwargs
            if _k != "seed_seq_list"
        }
    else:
        _sim_clrenf_cnts_ll_kwargs = {}

    _res_list = Parallel(n_jobs=_thread_count, prefer="threads")(
        delayed(sim_clrenf_cnts)(
            _clrenf_parm_vec,
            _mkt_sample_spec_here,
            **_sim_clrenf_cnts_ll_kwargs,
            saved_array_name_suffix=f"{_thread_id:0{int(np.ceil(np.log10(_thread_count)))}d}",
            seed_seq_list=_rng_seed_seq_list_ch,
        )
        for _thread_id, _rng_seed_seq_list_ch in enumerate(_rng_seed_seq_list)
    )

    _res_list_stacks = [np.stack([_j[_k] for _j in _res_list]) for _k in range(3)]
    (
        _clrenf_cnts_sim_byfirmcount_array,
        _clrenf_cnts_sim_bydelta_array,
        _clrenf_cnts_sim_byconczone_array,
    ) = (
        np.column_stack((
            _g[0, :, :_h],
            np.einsum("ijk->jk", np.int_(1) * _g[:, :, _h:]),
        ))
        for _g, _h in zip(_res_list_stacks, [1, 1, 3], strict=True)
    )
    del _res_list, _res_list_stacks

    return (
        _clrenf_cnts_sim_byfirmcount_array,
        _clrenf_cnts_sim_bydelta_array,
        _clrenf_cnts_sim_byconczone_array,
    )


def sim_clrenf_cnts(
    _guppi_test_parms: Sequence[float],
    _mkt_sample_spec: dgl.MarketSampleSpec,
    /,
    *,
    sim_clrenf_sel: tuple[
        clstl.CLRENFSelector, GUPPIWghtngSelector, GUPPIWghtngSelector | None
    ],
    saved_array_name_suffix: str = "",
    save_data_to_file: Literal[False] | tuple[Literal[True], ptb.File, str] = False,
    seed_seq_list: list[SeedSequence] | None = None,
    nthreads: int = 16,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    if _mkt_sample_spec.recapture_rate != _guppi_test_parms[0]:
        raise ValueError(
            "{} {} {} {}".format(
                f"Value, {_mkt_sample_spec.recapture_rate}",
                "of recapture rate in the second positional argument",
                f"must equal its value, {_guppi_test_parms[0]}",
                "in the first positional argument.",
            )
        )

    # Generate market data
    _market_data = dgl.gen_market_sample(
        _mkt_sample_spec, seed_seq_list=seed_seq_list, nthreads=nthreads
    )

    _invalid_array_names = (
        ("fcounts", "choice_prob_outgd", "nth_firm_share", "hhi_post")
        if _mkt_sample_spec.share_spec[0] == "Uniform"
        else ()
    )

    save_namedtuple_to_hdf5(
        _market_data,
        saved_array_name_suffix,
        _invalid_array_names,
        save_data_to_file=save_data_to_file,
    )

    _upp_tests_data = gen_guppi_arrays(
        _guppi_test_parms,
        _market_data,
        sim_clrenf_sel,
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
    _firm_counts_prob_weights = _mkt_sample_spec.share_spec[-1]
    _max_firm_count = (
        len(_firm_counts_prob_weights)
        if hasattr(_firm_counts_prob_weights, "__len__")
        else 9
    )
    _clrenf_cnts_sim_byfirmcount_array = -1 * np.ones(_stats_rowlen, np.int_)
    for _firm_cnt in 2 + np.arange(_max_firm_count):
        _firm_count_test = _fcounts == _firm_cnt

        _clrenf_cnts_sim_byfirmcount_array = np.row_stack((
            _clrenf_cnts_sim_byfirmcount_array,
            np.array([
                _firm_cnt,
                np.einsum("ij->", 1 * _firm_count_test),
                *[
                    np.einsum("ij->", 1 * (_firm_count_test & _f))
                    for _f in _upp_tests_data
                ],
            ]),
        ))
    _clrenf_cnts_sim_byfirmcount_array = _clrenf_cnts_sim_byfirmcount_array[1:]

    _hhi_delta_ranged = clstl.hhi_delta_ranger(_hhi_delta)
    _clrenf_cnts_sim_bydelta_array = -1 * np.ones(_stats_rowlen, np.int_)
    for _hhi_delta_lim in clstl.hhi_delta_knots[:-1]:
        _hhi_delta_test = _hhi_delta_ranged == _hhi_delta_lim

        _clrenf_cnts_sim_bydelta_array = np.row_stack((
            _clrenf_cnts_sim_bydelta_array,
            np.array([
                _hhi_delta_lim,
                np.einsum("ij->", 1 * _hhi_delta_test),
                *[
                    np.einsum("ij->", 1 * (_hhi_delta_test & _f))
                    for _f in _upp_tests_data
                ],
            ]),
        ))

    _clrenf_cnts_sim_bydelta_array = _clrenf_cnts_sim_bydelta_array[1:]

    # Clearance/enfrocement counts --- by zone, coded
    try:
        _hhi_zone_post_ranged = clstl.hhi_zone_post_ranger(_hhi_post)
    except ValueError as _err:
        print(_hhi_post)
        raise _err

    _stats_byconczone_sim = -1 * np.ones(_stats_rowlen + 1, np.int_)
    for _hhi_zone_post_knot in clstl.hhi_zone_post_knots[:-1]:
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

    _clrenf_cnts_sim_byconczone_array = clstl.clrenf_cnts_byconczone(
        _stats_byconczone_sim[1:]
    )
    del _stats_byconczone_sim
    del _hhi_delta, _hhi_post, _fcounts

    return (
        _clrenf_cnts_sim_byfirmcount_array,
        _clrenf_cnts_sim_bydelta_array,
        _clrenf_cnts_sim_byconczone_array,
    )


def gen_guppi_arrays(
    _guppi_test_parms: Sequence[float],
    _market_data: dgl.MarketSample,
    _sim_clrenf_sel: tuple[
        clstl.CLRENFSelector, GUPPIWghtngSelector, GUPPIWghtngSelector | None
    ],
    /,
    *,
    saved_array_name_suffix: str = "",
    save_data_to_file: Literal[False] | tuple[Literal[True], ptb.File, str] = False,
) -> UPPTests:
    _r_bar, _g_bar, _divr_bar, _cmcr_bar, _ipr_bar = _guppi_test_parms

    _clrenf_sel, _guppi_wgtng_policy, _divr_wgtng_policy = _sim_clrenf_sel

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
    np.divide(_guppi_array, (1 - _market_data.divr_array[:, ::-1]), out=_ipr_array)

    # _ipr_array_alt = np.empty_like(_market_data.divr_array)
    # np.divide(
    #     np.einsum("ij,ij->ij", _market_data.pcm_array, _market_data.divr_array),
    #     1 - _market_data.divr_array,
    #     out=_ipr_array_alt,
    # )

    _test_measure_seq = (_market_data.divr_array, _guppi_array, _cmcr_array, _ipr_array)

    _wt_array = (
        _market_data.frmshr_array / np.einsum("ij->i", _market_data.frmshr_array)
        if _guppi_wgtng_policy
        in (
            GUPPIWghtngSelector.CPA,
            GUPPIWghtngSelector.CPD,
            GUPPIWghtngSelector.OSA,
            GUPPIWghtngSelector.OSD,
        )
        else dgl.DEFAULT_EMPTY_ARRAY
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

    if _clrenf_sel == clstl.CLRENFSelector.ENFT:
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
    save_data_to_file: Literal[False] | tuple[Literal[True], ptb.File, str] = False,
) -> None:
    match save_data_to_file:
        case False:
            _save_data_to_file_flag = False
        case True, _h5_datafile, _h5_hier:
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
        case _:
            print(
                'Object, "save_data_to_file" must be one of',
                "(False, (True, <PyTables file handle>, <HDF5 hierarchy>),)",
            )
            raise ValueError


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
    print(f"PyTables: Now saving array, {_h5_array_name!r}, in subpath, {_h5_hier!r}")
    _h5_array = _h5_datafile.create_carray(
        _h5_hier,
        _h5_array_name,
        obj=_data_array,
        createparents=True,
        title=f"{_array_name})",
    )
    _h5_array[:] = _data_array
