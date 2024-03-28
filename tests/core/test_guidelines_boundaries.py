import gc
from collections.abc import Sequence

import mergeron.core.guidelines_boundaries as gbl
import mergeron.core.guidelines_boundary_functions as gbfn
import mergeron.core.guidelines_boundary_functions_extra as gbxtr
import pytest
from mergeron import RECConstants, UPPAggrSelector
from mergeron.core import UPPBoundarySpec
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
    assert_equal(gbfn.round_cust(_num, frac=_frac, rounding_mode=_mode), _test_val)


def test_round_cust_to_fp_aproximation_error() -> None:
    # Difference below is due to floating-point approx. error
    assert_almost_equal(
        gbfn.round_cust(12.35, frac=0.05, rounding_mode="ROUND_DOWN"), 12.35, decimal=12
    )


def test_lerp() -> None:
    assert_equal(gbfn.lerp(1, 3, 0.25), 1.5)


@pytest.mark.parametrize(
    "_dhv, _rbar, _test_val",
    (
        (0.01, 4 / 5, 0.06),
        (0.02, 4 / 5, 0.09),
        (0.005, 6 / 7, 0.045),
        (0.01, 6 / 7, 0.065),
    ),
)
def test_guppi_from_delta(_dhv: float, _rbar: float, _test_val: float) -> None:
    assert_equal(gbl.guppi_from_delta(_dhv, r_bar=_rbar), _test_val)


@pytest.mark.parametrize(
    "_gv, _mv, _rv, _test_val", ((0.06, 1.00, 4 / 5, 0.070), (0.09, 0.40, 0.9, 0.200))
)
def test_share_from_guppi(_gv: float, _mv: float, _rv: float, _test_val: float) -> None:
    assert_equal(gbl.share_from_guppi(_gv, m_star=_mv, r_bar=_rv), _test_val)


@pytest.mark.parametrize(
    "_test_parms, _test_val",
    zip(((), (0.045, 1.00, 6 / 7)), (0.075, 5 / 95), strict=True),
)
def test_benchmark_shrratio(_test_parms: Sequence[float], _test_val: float) -> None:
    if _test_parms:
        _gv, _mv, _rv = _test_parms
        _ts = gbl.critical_share_ratio(_gv, m_star=_mv, r_bar=_rv)
    else:
        _ts = gbl.critical_share_ratio()
    assert_equal(gbfn.round_cust(_ts), gbfn.round_cust(_test_val))


def print_done() -> None:
    print("... done.")


_dh_tuple = ((0.01, 0.03147), (0.02, 0.05595), (0.08, 0.16709))


@pytest.mark.parametrize("_dhv, _dha", _dh_tuple)
def test_dh_area(_dhv: float, _dha: float) -> None:
    print(f"Testing gbl.dh_area() with ΔHHI value of {_dhv} ... ", end="")
    try:
        assert_equal(gbfn.dh_area(_dhv), gbxtr.dh_area_quad(_dhv))
    except AssertionError as _err:
        print(gbfn.dh_area(_dhv), "=?", gbxtr.dh_area_quad(_dhv), end="")
        raise _err
    print_done()


@pytest.mark.parametrize("_dhv, _dha", _dh_tuple)
def test_hhi_delta_boundary_dha(_dhv: float, _dha: float) -> None:
    _ts = gbl.hhi_delta_boundary(_dhv).area
    print(f"Testing gbl.hhi_delta_boundary() with ΔHHI value of {_dhv} ... ", end="")
    try:
        assert_equal(_ts, _dha)
    except AssertionError as _err:
        print(gbfn.dh_area(_dhv), "=?", _ts, end="")
        raise _err
    print_done()


@pytest.mark.parametrize("_dhv", (0.01, 0.02, 0.08))
def test_hhi_delta_boundary(_dhv: float) -> None:
    _ts = gbl.hhi_delta_boundary(_dhv).area
    print(f"Testing gbl.hhi_delta_boundary() with ΔHHI value of {_dhv} ... ", end="")
    try:
        assert_equal(_ts, round(gbfn.dh_area(_dhv), 5))
    except AssertionError as _err:
        print(gbfn.dh_area(_dhv), "=?", _ts, end="")
        raise _err
    print_done()


@pytest.mark.parametrize("_dhv", (0.02, 0.0625, 0.16))
def test_combined_share_boundary(_dhv: float) -> None:
    assert_equal(gbl.combined_share_boundary(mp.sqrt(_dhv)).area, _dhv / 2)


@pytest.mark.parametrize("_dhv", (0.02, 0.03125, 0.08))
def test_hhi_pre_contrib_boundary(_dhv: float) -> None:
    assert_equal(
        gbl.hhi_pre_contrib_boundary(_dhv).area, round(mp.pi * mpf(f"{_dhv}") / 4, 5)
    )


@pytest.mark.parametrize(
    "_dhv, _tv",
    zip(
        ((0.06, 1.00), (0.06, 0.67), (0.06, 0.30)),
        (0.0052325581, 0.0112691576, 0.05),
        strict=True,
    ),
)
def test_diversion_ratio_boundary_at_max(_dhv: tuple[float, float], _tv: float) -> None:
    _test_area = gbl.diversion_ratio_boundary(
        UPPBoundarySpec(
            gbl.critical_share_ratio(_dhv[0], m_star=_dhv[1], r_bar=1.00),
            0.80,
            agg_method=UPPAggrSelector.MAX,
            precision=10,
        )
    ).area
    assert_equal(_test_area, _tv)


@pytest.mark.parametrize(
    "_dhv, _tv",
    zip(
        ((0.06, 1.00), (0.06, 0.67), (0.06, 0.30)),
        (0.0052325581, 0.0112691576, 0.05),
        strict=True,
    ),
)
def test_shrratio_boundary_max(_dhv: tuple[float, float], _tv: float) -> None:
    _test_area = gbfn.shrratio_boundary_max(
        gbl.critical_share_ratio(_dhv[0], m_star=_dhv[1], r_bar=0.80)
    ).area
    assert_equal(_test_area, _tv)


@pytest.mark.parametrize(
    "_gv, _mv, _rv, _recapture_form",
    (
        (0.06, 1.00, 0.8, RECConstants.FIXED),
        (0.06, 0.67, 0.8, RECConstants.FIXED),
        (0.06, 0.30, 0.8, RECConstants.FIXED),
    ),
)
def test_shrratio_boundary_at_min(
    _gv: float, _mv: float, _rv: float, _recapture_form: RECConstants
) -> None:
    _test_area = gbl.diversion_ratio_boundary(
        UPPBoundarySpec(
            gbl.critical_share_ratio(_gv, m_star=_mv, r_bar=1.00),
            _rv,
            agg_method=UPPAggrSelector.MIN,
            recapture_form=_recapture_form,
            precision=10,
        )
    ).area
    assert_equal(
        gbfn.round_cust(_test_area), gbl.share_from_guppi(_gv, m_star=_mv, r_bar=_rv)
    )


@pytest.mark.parametrize(
    "_gv, _mv, _rv, _recapture_form",
    (
        (0.06, 1.00, 0.8, "proportional"),
        (0.06, 0.67, 0.8, "proportional"),
        (0.06, 0.30, 0.8, "proportional"),
    ),
)
def test_shrratio_boundary_min(
    _gv: float, _mv: float, _rv: float, _recapture_form: str
) -> None:
    assert_equal(
        gbfn.round_cust(
            gbfn.shrratio_boundary_min(
                gbl.critical_share_ratio(_gv, m_star=_mv, r_bar=_rv),
                _rv,
                recapture_form=_recapture_form,
            ).area
        ),
        gbl.share_from_guppi(_gv, m_star=_mv, r_bar=_rv),
    )


@pytest.mark.parametrize(
    "_tvl",
    (
        (0.06, 1.0, UPPAggrSelector.OSA, RECConstants.FIXED, 0.05111),
        (0.06, 0.67, UPPAggrSelector.OSA, RECConstants.FIXED, 0.07726),
        (0.06, 0.3, UPPAggrSelector.OSA, RECConstants.FIXED, 0.17098),
        (0.06, 1.0, UPPAggrSelector.CPA, RECConstants.FIXED, 0.00658),
        (0.06, 0.67, UPPAggrSelector.CPA, RECConstants.FIXED, 0.01409),
        (0.06, 0.3, UPPAggrSelector.CPA, RECConstants.FIXED, 0.0606),
        (0.06, 1.0, UPPAggrSelector.AVG, RECConstants.FIXED, 0.0102),
        (0.06, 0.67, UPPAggrSelector.AVG, RECConstants.FIXED, 0.02169),
        (0.06, 0.3, UPPAggrSelector.AVG, RECConstants.FIXED, 0.09123),
        (0.06, 1.0, UPPAggrSelector.AVG, RECConstants.INOUT, 0.01026),
        (0.06, 0.67, UPPAggrSelector.AVG, RECConstants.INOUT, 0.02187),
        (0.06, 0.3, UPPAggrSelector.AVG, RECConstants.INOUT, 0.09323),
    ),
)
def test_diversion_ratio_boundary(_tvl: tuple[float, float, str, str, float]) -> None:
    _bdry_spec = UPPBoundarySpec(
        gbl.critical_share_ratio(_tvl[0], m_star=_tvl[1], r_bar=1.0),
        0.80,
        agg_method=_tvl[2],  # type: ignore
        recapture_form=_tvl[3],  # type: ignore
    )
    _ts = gbl.diversion_ratio_boundary(_bdry_spec).area
    print("Test gbl.diversion_ratio_boundary_wtd_avg(): ", end="")
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


shrratio_boundary_wtd_avg_test_values = (
    (0.06, 1.0, "own-share", "arithmetic", "proportional", 0.05111),
    (0.06, 0.67, "own-share", "arithmetic", "proportional", 0.07726),
    (0.06, 0.3, "own-share", "arithmetic", "proportional", 0.17098),
    (0.06, 1.0, "cross-product-share", "arithmetic", "proportional", 0.00658),
    (0.06, 0.67, "cross-product-share", "arithmetic", "proportional", 0.01409),
    (0.06, 0.3, "cross-product-share", "arithmetic", "proportional", 0.0606),
    (0.06, 1.0, None, "arithmetic", "proportional", 0.0102),
    (0.06, 0.67, None, "arithmetic", "proportional", 0.02169),
    (0.06, 0.3, None, "arithmetic", "proportional", 0.09123),
    (0.06, 1.0, None, "arithmetic", "inside-out", 0.01026),
    (0.06, 0.67, None, "arithmetic", "inside-out", 0.02187),
    (0.06, 0.3, None, "arithmetic", "inside-out", 0.09323),
)


@pytest.mark.parametrize("_tvl", shrratio_boundary_wtd_avg_test_values)
def test_shrratio_boundary_wtd_avg(_tvl: tuple[float, float, str, str, float]) -> None:
    _ts = gbfn.shrratio_boundary_wtd_avg(
        gbl.critical_share_ratio(_tvl[0], m_star=_tvl[1], r_bar=0.80),
        0.80,
        weighting=_tvl[2],  # type: ignore
        agg_method=_tvl[3],  # type: ignore
        recapture_form=_tvl[4],  # type: ignore
    ).area
    print("Test gbl.shrratio_boundary_wtd_avg(): ", end="")
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
def test_shrratio_boundary_xact_avg(_tvl: tuple[float, float, str, float]) -> None:
    _ts = gbfn.shrratio_boundary_xact_avg(
        gbl.critical_share_ratio(_tvl[0], m_star=_tvl[1], r_bar=0.80),
        0.80,
        recapture_form=_tvl[2],  # type: ignore
    ).area
    print("Test gbl.gen_xact_avg_shrratio_mgnsym_boundary(): ", end="")
    try:
        assert_equal(_ts, _tvl[-1])
    except AssertionError as _err:
        print(gval_print_format_str.format(*_tvl, _ts), end="")
        raise _err
    print_done()


@pytest.mark.parametrize(
    "_tvl",
    (
        (0.06, 1.0, "own-share", "proportional", 0.05109304376203841),
        (0.06, 0.67, "own-share", "proportional", 0.07725625014417284),
        (0.06, 0.3, "own-share", "proportional", 0.17095706110994838),
        (0.06, 1.0, "cross-product-share", "proportional", 0.006600706829415734),
        (0.06, 0.67, "cross-product-share", "proportional", 0.01409071869102121),
        (0.06, 0.3, "cross-product-share", "proportional", 0.0606003596099902),
        (0.06, 1.0, None, "proportional", 0.010202305341592572711721127),
        (0.06, 0.67, None, "proportional", 0.021688811233381205151727031),
        (0.06, 0.3, None, "proportional", 0.091226363716114866214444818),
        (0.06, 1.0, None, "inside-out", 0.010256625940293613886783542),
        (0.06, 0.67, None, "inside-out", 0.021867653765402224297618528),
        (0.06, 0.3, None, "inside-out", 0.093231884646380737814431487),
    ),
)
def test_shrratio_boundary_qdtr_wtd_avg(
    _tvl: tuple[float, float, str, str, float],
) -> None:
    _ts = gbxtr.shrratio_boundary_qdtr_wtd_avg(
        gbl.critical_share_ratio(_tvl[0], m_star=_tvl[1], r_bar=0.80),
        0.80,
        weighting=_tvl[2],  # type: ignore
        recapture_form=_tvl[3],  # type: ignore
    ).area
    print("Test gbxtr.shrratio_boundary_qdtr_wtd_avg(): ", end="")
    try:
        assert_equal(float(_ts), _tvl[-1])
    except AssertionError as _err:
        print(
            "g_val = {}; m_val = {}; wgtng = {}; meanf = {}; {}".format(*_tvl),
            "=?",
            _ts,
            end="",
        )
        raise _err
    print_done()


@pytest.mark.parametrize("_tvl", shrratio_boundary_wtd_avg_test_values)
def test_shrratio_boundary_distance(_tvl: tuple[float, float, str, str, float]) -> None:
    _ts = gbxtr.shrratio_boundary_distance(
        gbl.critical_share_ratio(_tvl[0], m_star=_tvl[1], r_bar=0.80),
        0.80,
        weighting=_tvl[2],  # type: ignore
        agg_method=_tvl[3],  # type: ignore
        recapture_form=_tvl[4],  # type: ignore
    ).area
    print("Test gbxtr.test_shrratio_boundary_distance(): ", end="")
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


gc.collect()
