import gc
from collections.abc import Sequence

import mergeron.core.guidelines_standards as gsf
import pytest
from mpmath import mp, mpf  # type: ignore
from numpy.testing import assert_almost_equal, assert_equal

gval_print_format_str = "g_val = {}; m_val = {}; {} =? {}"


@pytest.mark.parametrize(
    "_num, _frac, _mode, _test_val",
    (
        (0.0624, 0.005, "ROUND_HALF_UP", 0.06),
        (8.8888, 0.50, "ROUND_DOWN", 8.5),
        (8.8888, 0.50, "ROUND_HALF_UP", 9.0),
    ),
)
def test_round_cust(_num: float, _frac: float, _mode: str, _test_val: float) -> None:
    assert_equal(gsf.round_cust(_num, frac=_frac, mode=_mode), _test_val)


def test_round_cust_to_fp_aproximation_error() -> None:
    # Difference below is due to floating-point approx. error
    assert_almost_equal(
        gsf.round_cust(12.35, frac=0.05, mode="ROUND_DOWN"), 12.35, decimal=12
    )


def test_lerp() -> None:
    assert_equal(gsf.lerp(1, 3, 0.25), 1.5)


@pytest.mark.parametrize(
    "_dhv, _rbar, _test_val",
    (
        (0.01, 4 / 5, 0.06),
        (0.02, 4 / 5, 0.09),
        (0.005, 6 / 7, 0.045),
        (0.01, 6 / 7, 0.065),
    ),
)
def test_gbd_from_dsf(_dhv: float, _rbar: float, _test_val: float) -> None:
    assert_equal(gsf.gbd_from_dsf(_dhv, r_bar=_rbar), _test_val)


@pytest.mark.parametrize(
    "_gv, _mv, _rv, _test_val", ((0.06, 1.00, 4 / 5, 0.070), (0.09, 0.40, 0.9, 0.200))
)
def test_shr_from_gbd(_gv: float, _mv: float, _rv: float, _test_val: float) -> None:
    assert_equal(gsf.shr_from_gbd(_gv, m_star=_mv, r_bar=_rv), _test_val)


@pytest.mark.parametrize(
    "_test_parms, _test_val",
    zip(((), (0.045, 1.00, 6 / 7)), (0.075, 5 / 95), strict=True),
)
def test_benchmark_shrratio(_test_parms: Sequence[float], _test_val: float) -> None:
    assert_equal(
        gsf.round_cust(gsf.critical_shrratio(*_test_parms)), gsf.round_cust(_test_val)
    )


def print_done() -> None:
    print("... done.")


_dh_tuple = ((0.01, 0.03147), (0.02, 0.05595), (0.08, 0.16709))


@pytest.mark.parametrize("_dhv, _dha", _dh_tuple)
def test_dh_area(_dhv: float, _dha: float) -> None:
    print(f"Testing gsf.dh_area() with ΔHHI value of {_dhv} ... ", end="")
    try:
        assert_equal(gsf.dh_area(_dhv), gsf.dh_area_quad(_dhv))
    except AssertionError as _err:
        print(gsf.dh_area(_dhv), "=?", gsf.dh_area_quad(_dhv), end="")
        raise _err
    print_done()


@pytest.mark.parametrize("_dhv, _dha", _dh_tuple)
def test_delta_hhi_boundary_dha(_dhv: float, _dha: float) -> None:
    _ts = gsf.delta_hhi_boundary(_dhv)[1]
    print(f"Testing gsf.delta_hhi_boundary() with ΔHHI value of {_dhv} ... ", end="")
    try:
        assert_equal(_ts, _dha)
    except AssertionError as _err:
        print(gsf.dh_area(_dhv), "=?", _ts, end="")
        raise _err
    print_done()


@pytest.mark.parametrize("_dhv", (0.01, 0.02, 0.08))
def test_delta_hhi_boundary(_dhv: float) -> None:
    _ts = gsf.delta_hhi_boundary(_dhv)[1]
    print(f"Testing gsf.delta_hhi_boundary() with ΔHHI value of {_dhv} ... ", end="")
    try:
        assert_equal(_ts, round(gsf.dh_area(_dhv), 5))
    except AssertionError as _err:
        print(gsf.dh_area(_dhv), "=?", _ts, end="")
        raise _err
    print_done()


@pytest.mark.parametrize("_dhv", (0.02, 0.0625, 0.16))
def test_combined_share_boundary(_dhv: float) -> None:
    assert_equal(gsf.combined_share_boundary(mp.sqrt(_dhv))[1], _dhv / 2)


@pytest.mark.parametrize("_dhv", (0.02, 0.03125, 0.08))
def test_hhi_pre_contrib_boundary(_dhv: float) -> None:
    assert_equal(
        gsf.hhi_pre_contrib_boundary(_dhv)[1], round(mp.pi * mpf(f"{_dhv}") / 4, 5)
    )


@pytest.mark.parametrize(
    "_dhv, _tv",
    zip(
        ((0.06, 1.00), (0.06, 0.67), (0.06, 0.30)),
        (0.0052325581, 0.0112691576, 0.05),
        strict=True,
    ),
)
def test_shrratio_mgnsym_boundary_max(_dhv: tuple[float, float], _tv: float) -> None:
    assert_equal(gsf.shrratio_mgnsym_boundary_max(gsf.critical_shrratio(*_dhv))[1], _tv)


@pytest.mark.parametrize(
    "_gv, _mv, _rv, _recapture_spec",
    (
        (0.06, 1.00, 0.8, "proportional"),
        (0.06, 0.67, 0.8, "proportional"),
        (0.06, 0.30, 0.8, "proportional"),
    ),
)
def test_shrratio_mgnsym_boundary_min(
    _gv: float, _mv: float, _rv: float, _recapture_spec: str
) -> None:
    def _s_from_d(x: float) -> mpf:
        return mp.fdiv(x, mp.fadd(1.0, x))

    assert_equal(
        gsf.round_cust(
            gsf.shrratio_mgnsym_boundary_min(
                gsf.critical_shrratio(_gv, _mv, _rv),
                _rv,
                recapture_spec=_recapture_spec,
            )[1]
        ),
        # round(_s_from_d(mp.fdiv(f"{_gv}", mp.fmul(f"{_mv}", f"{_rv}"))), 10),
        gsf.shr_from_gbd(_gv, m_star=_mv, r_bar=_rv),
    )


@pytest.mark.parametrize(
    "_tvl",
    (
        (0.06, 1.0, "own-share", "arithmetic", 0.05092),
        (0.06, 0.67, "own-share", "arithmetic", 0.07697),
        (0.06, 0.3, "own-share", "arithmetic", 0.17016),
        (0.06, 1.0, "cross-product-share", "arithmetic", 0.00658),
        (0.06, 0.67, "cross-product-share", "arithmetic", 0.01409),
        (0.06, 0.3, "cross-product-share", "arithmetic", 0.0606),
        (0.06, 1.0, "own-share", "geometric", 0.05253),
        (0.06, 0.67, "own-share", "geometric", 0.07903),
        (0.06, 0.3, "own-share", "geometric", 0.17263),
        (0.06, 1.0, "cross-product-share", "geometric", 0.00661),
        (0.06, 0.67, "cross-product-share", "geometric", 0.01417),
        (0.06, 0.3, "cross-product-share", "geometric", 0.06121),
    ),
)
def test_shrratio_mgnsym_boundary_wtd_avg(
    _tvl: tuple[float, float, str, str, float],
) -> None:
    _ts = gsf.shrratio_mgnsym_boundary_wtd_avg(
        gsf.critical_shrratio(_tvl[0], _tvl[1]),
        wgtng_policy=_tvl[2],  # type: ignore
        avg_method=_tvl[3],  # type: ignore
        recapture_spec="proportional",
    )[1]
    print("Test gsf.shrratio_mgnsym_boundary_wtd_avg(): ", end="")
    try:
        assert_equal(_ts, _tvl[-1])
    except AssertionError as _err:
        print(
            "g_val = {}; m_val = {}; wgtng = {}; meanf = {}; {}".format(*_tvl),
            "=?",
            _ts,
            end="",
        )
        raise _err
    print_done()


@pytest.mark.parametrize(
    "_tvl",
    (
        (0.06, 1.0, "inside-out", 0.01026),
        (0.06, 0.67, "inside-out", 0.02187),
        (0.06, 0.3, "inside-out", 0.09323),
        (0.06, 1.0, "proportional", 0.0102),
        (0.06, 0.67, "proportional", 0.02169),
        (0.06, 0.3, "proportional", 0.09123),
    ),
)
def test_shrratio_mgnsym_boundary_xact_avg(
    _tvl: tuple[float, float, str, float],
) -> None:
    _ts = gsf.shrratio_mgnsym_boundary_xact_avg(
        gsf.critical_shrratio(_tvl[0], _tvl[1]),
        recapture_spec=_tvl[2],  # type: ignore
    )[1]
    print("Test gsf.gen_xact_avg_shrratio_mgnsym_boundary(): ", end="")
    try:
        assert_equal(_ts, _tvl[-1])
    except AssertionError as _err:
        print(gval_print_format_str.format(*_tvl, _ts), end="")
        raise _err
    print_done()


@pytest.mark.parametrize(
    "_tvl",
    (
        (0.06, 1.0, "inside-out", 0.01026),
        (0.06, 0.67, "inside-out", 0.02188),
        (0.06, 0.3, "inside-out", 0.09324),
        (0.06, 1.0, "proportional", 0.01021),
        (0.06, 0.67, "proportional", 0.0217),
        (0.06, 0.3, "proportional", 0.09124),
    ),
)
def test_shrratio_mgnsym_boundary_avg(_tvl: tuple[float, float, str, float]) -> None:
    _ts = gsf.shrratio_mgnsym_boundary_avg(
        gsf.critical_shrratio(_tvl[0], _tvl[1]),
        recapture_spec=_tvl[2],  # type: ignore
    )[1]
    print("Test gsf.shrratio_mgnsym_boundary_avg(): ", end="")
    try:
        assert_equal(_ts, _tvl[-1])
    except AssertionError as _err:
        print(gval_print_format_str.format(*_tvl, _ts), end="")
        raise _err
    print_done()


gc.collect()
