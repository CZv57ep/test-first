"""

    Estimation of intrinsic clearance rates for ΔHHI safeharbor and GUPPI safeharbor
    Demonstrates MC integration with closed-form integrand and indicator-function
    integrand (Tested with Python 3.8 and 3.9; NumPy module is required.)

    S. Murthy Kambhampaty, © 2019. This work is licensed under a CC BY-NC-SA 4.0 License
    https://creativecommons.org/licenses/by-nc-sa/4.0/

"""

from datetime import datetime

import numpy as np
from numpy.random import PCG64DXSM, Generator
from numpy.typing import NDArray

rng = Generator(PCG64DXSM())

dh_bar = 0.01
g_bar = 0.06
r_bar = 0.80
d_bar = 0.2  # 0.2  # r_bar  # g_bar  # 0.8 * 0.125 / (1 - 0.125)
sample_sz = 10**8


def icr_gsh_sym(_ssz: int = sample_sz) -> NDArray[float]:
    """With symmetric shares, margins, and prices; closed-form integrand."""
    _m_lim = g_bar / d_bar
    _m_star = _m_lim + (1.0 - _m_lim) * rng.uniform(size=(_ssz, 1))
    _d_star = g_bar / (r_bar * _m_star)
    _divr_limit_prob = 2 * g_bar / (r_bar + d_bar)
    _gsf_bound_prob = 2 * (1 - _m_lim) * (_d_star / (1 + _d_star)).mean()
    return np.array([_ssz, _divr_limit_prob + _gsf_bound_prob, np.nan, np.nan])


def icr_gsh_asymmshr(_ssz: int = sample_sz) -> NDArray[float]:
    """With symmetric margins and prices, unequal shares; closed-form integrand."""
    _m_lim = g_bar / d_bar
    _m_star = _m_lim + (1.0 - _m_lim) * rng.random(size=(_ssz, 1))

    _d_star = g_bar / (r_bar * _m_star)
    _divr_limit_prob = 2 * (g_bar / r_bar) * d_bar / (r_bar + d_bar)
    _gsf_bound_prob = 2 * (1 - _m_lim) * (_d_star**2 / (1 + _d_star)).mean()
    return np.array([_ssz, _divr_limit_prob + _gsf_bound_prob, np.nan, np.nan])


def gen_qtyshr(_ssz: int = sample_sz, *, sym_flag: bool = False) -> NDArray[float]:
    """Unequal shares and margins, and symmetric prices; indicator-function integrand."""
    if sym_flag:
        # for symmetric shares
        _mktshr_array = 0.5 * rng.uniform(size=(_ssz, 1))
        _mktshr_array = _mktshr_array[:, [0, 0]]
    else:
        _mktshr_array = np.sort(rng.random(size=(_ssz, 2)))
        _mktshr_array = np.column_stack((_mktshr_array[:, [0]], np.diff(_mktshr_array)))
    return _mktshr_array


def icr_gsh_asymmshrmgn(_ssz: int = sample_sz) -> NDArray[float]:
    """With symmetric prices, unequal shares and margins; indicator-function integrand."""

    _shr_sym_flag = False
    _mktshr_array = gen_qtyshr(_ssz, sym_flag=_shr_sym_flag)
    _hhi_delta = np.einsum("ij,ij->i", _mktshr_array, _mktshr_array[:, ::-1])
    _divr_array = r_bar * _mktshr_array[:, ::-1] / (1 - _mktshr_array)
    _delta_test = _hhi_delta < dh_bar
    del _hhi_delta

    _pcm_array = rng.uniform(size=_divr_array.shape)
    _pcm_sym_flag = False
    if _pcm_sym_flag:
        _pcm_array = _pcm_array[:, [0, 0]]

    _pr_sym_flag = True
    _pr_corr_sign = None
    if _pr_corr_sign not in (_pcvs := (None, "positive", "negative")):
        raise ValueError(f"Price correlation must be one of {_pcvs!r}")
    if _pr_sym_flag:
        _pr_ratio_array = np.ones(_pcm_array.shape)
    else:
        _pr_max_ratio = 5
        if _pr_corr_sign == "positive":
            _pr_array = 1 + np.floor(_pr_max_ratio * _mktshr_array)
        elif _pr_corr_sign == "negative":
            _pr_array = _pr_max_ratio - np.floor(_pr_max_ratio * _mktshr_array)
        else:
            _pr_array = rng.choice(
                1.00 + np.arange(_pr_max_ratio), size=_pcm_array.shape
            )
        _pr_ratio_array = np.divide(_pr_array, _pr_array[:, ::-1])
        del _pr_max_ratio, _pr_array

    _guppi_array = np.einsum(
        "ij,ij,ij->ij", _divr_array, _pcm_array[:, ::-1], _pr_ratio_array[:, ::-1]
    )
    _gbd_test = _guppi_array.max(axis=1) < g_bar
    _divr_test = _divr_array.max(axis=1) < d_bar
    _pcm_min_test = _pcm_array.min(axis=1) >= (g_bar / d_bar)
    _divr_limit_test = _gbd_test & _divr_test & np.logical_not(_pcm_min_test)
    _gsf_not_in_deltah = np.logical_not(_delta_test) & _gbd_test & _divr_test

    _scount = len(_gbd_test)
    _gsf_prob = len(_gbd_test[_gbd_test & _divr_test]) / _scount
    _deltah_prob = len(_gbd_test[_delta_test]) / _scount
    _cum_clr_prob = len(_gbd_test[_delta_test | _gsf_not_in_deltah]) / _scount

    del _guppi_array, _divr_array, _pcm_array
    del _gbd_test, _divr_test, _pcm_min_test, _divr_limit_test

    return np.array([_scount, _gsf_prob, _deltah_prob, _cum_clr_prob])


if __name__ == "__main__":
    for run_func in (icr_gsh_sym, icr_gsh_asymmshr, icr_gsh_asymmshrmgn):
        resv = np.array([0.0, 0.0, 0.0, 0.0])
        icount = 1000
        stime = datetime.now()
        for _ in range(icount):
            tmpv = run_func(sample_sz // icount)
            resv += tmpv[0] * tmpv
        resv = np.round(resv / (icount * np.sqrt(resv[0] / icount)), 4)

        # In the printed output,
        #     first column reports the intrinsic clearance rate for
        #       the GUPPI safeharbor Where reported,
        #     second column reports the intrinsic clearance rate for the ΔHHI safeharbor
        #     third column reports the intrinsic clearance rate for cumulative application
        #         of the ΔHHI safeharbor and the GUPPI safeharbor
        print(
            np.sum(resv[1]),
            resv[-2],
            resv[-1],
            f"; duration {(datetime.now() - stime).total_seconds()} secs.",
        )
