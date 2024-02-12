from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal

import mergeron.core.proportions_tests as pci
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.typing import NDArray

test_propn_ci_cases = MappingProxyType({
    "Wilson": (0.4, 0.427753279986, 0.168180329706, 0.687326230266),
    "Agresti-Coull": (0.4, 6.0 / 14.0, 0.169346564040, 0.687796293102),
    "Clopper-Pearson": (0.4, 0.429587090756, 0.121552258120, 0.737621923393),
})


@pytest.mark.parametrize("_method", tuple(test_propn_ci_cases.keys()))
def test_propn_ci(
    _method: Literal["Agresti-Coull", "Clopper-Pearson", "Exact", "Wilson", "Score"],
) -> None:
    _test_val = test_propn_ci_cases[_method]
    _est_val = pci.propn_ci(4, 10, alpha=0.05, method=_method)
    try:
        assert_array_almost_equal(_est_val, _test_val, decimal=12)
    except AssertionError as _err:
        _err_str = (
            "Testing poportion c.i.s using method, "
            f"{_method!r}, test: {_test_val} ?? est: {_est_val}"
        )
        raise ValueError(_err_str) from _err


propn_diff_ci_test_cases: Mapping[str, tuple[tuple[int | float, ...], ...]] = (
    MappingProxyType({
        "counts": (
            # Comparison to R
            (3, 29, 6, 36),
            # Miettinen-Nurminen (1985), Example 5
            (0, 10, 0, 20),
            (10, 10, 20, 20),
            # Examples from Newcombe (1998), Table II
            (56, 70, 48, 80),
            (9, 10, 3, 10),
            (6, 7, 2, 7),
            (5, 56, 0, 29),
            (0, 10, 0, 20),
            (0, 10, 0, 10),
            (10, 10, 0, 20),
            (10, 10, 0, 10),
        ),
        "Agresti-Caffo": (
            (-0.225818, 0.1154615),
            (-0.14109010, 0.21684767),
            (-0.21684767, 0.14109010),
            (0.05245293, 0.3357585),
            (0.1600008, 0.8399992),
            (0.03380627, 0.8550826),
            (-0.02886585, 0.1712463),
            (-0.1410901, 0.2168477),
            (-0.2211503, 0.2211503),
            (0.6922432, 1.0),
            (0.612183, 1.0),
        ),
        "Newcombe": (
            (-0.2298067, 0.1197196),
            (-0.16112516, 0.27753280),
            (-0.27753280, 0.16112516),
            (0.0524, 0.3339),
            (0.1705, 0.8090),
            (0.0582, 0.8062),
            (-0.0381, 0.1926),
            (-0.1611, 0.2775),
            (-0.2775, 0.2775),
            (0.6791, 1.0),
            (0.6075, 1.0),
        ),
        "M-N": (
            (-0.2369807, 0.1227832),
            (-0.16576023, 0.28438134),
            (-0.28438134, 0.16576023),
            (0.0528, 0.3382),
            (0.1700, 0.8406),
            (0.0342, 0.8534),
            (-0.0326, 0.1933),
            (-0.1658, 0.2844),
            (-0.2879, 0.2879),
            (0.7156, 1.0),
            (0.6636, 1.0),
        ),
        "Mee": (
            (-0.2355, 0.1211),
            (-0.16112516, 0.27753279),
            (-0.27753279, 0.16112516),
            (0.0533, 0.3377),
            (0.1821, 0.8370),
            (0.0544, 0.8478),
            (-0.0313, 0.1926),
            (-0.1611, 0.2775),
            (-0.2775, 0.2775),
            (0.7225, 1.0),
            (0.6777, 1.0),
        ),
    })
)
propn_diff_ci_test_cases_tuple: tuple[
    tuple[str, int, Sequence[int | float], Sequence[int | float]], ...
] = ()
for _k in propn_diff_ci_test_cases:
    if _k == "counts":
        continue
    for _idx, (_counts_i, _contrast_cis_i) in enumerate(
        zip(
            propn_diff_ci_test_cases["counts"],
            propn_diff_ci_test_cases[_k],
            strict=True,
        )
    ):
        propn_diff_ci_test_cases_tuple += ((_k, _idx, _counts_i, _contrast_cis_i),)


@pytest.mark.parametrize("_method, _idx, _counts, _cis", propn_diff_ci_test_cases_tuple)
def test_propn_diff_ci(
    _method: Literal["Agresti-Caffo", "Mee", "M-N", "Newcombe", "Score"],
    _idx: int,
    _counts: Sequence[int],
    _cis: Sequence[float],
) -> None:
    if _idx == 0:
        num_digits = 4 if _method == "Mee" else 7

    elif _idx in (1, 2):
        num_digits = 8

    else:
        num_digits = 4

    _test_val = _cis
    _est_val = pci.propn_diff_ci(*_counts, method=_method)[-2:]

    try:
        assert_array_almost_equal(_est_val, _test_val, decimal=num_digits)

    except AssertionError as _err:
        _err_str = (
            "Estimated c.i. for difference in proportions "
            f"using method of {_method}, did not match test value: "
            f"counts: {_counts}; test: {_test_val} ?? est: {_est_val}"
        )
        raise ValueError(_err_str) from _err


def ques_data() -> Mapping[str, NDArray[np.int64]]:
    """
    Data from Quisenberry and Hurst, published in Goodman, 1965

    Returns
    -------
    dict: failure data as dict

    """

    _table_1_str = """
            Mode of Failure, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            Frequency, 5, 11, 19, 30, 58, 67, 92, 118, 173, 297
            """
    _t1_dat = [f.split(",") for f in _table_1_str.split("\n")[1:-1]]
    return {
        f[0].strip(): np.array([int(g) for g in f[1:]], dtype=np.int64) for f in _t1_dat
    }


@pytest.mark.parametrize(
    "_g_method, _g_alt, _t_val",
    zip(
        ("goodman", "quesenberry-hurst", "goodman", "quesenberry-hurst"),
        ("default", "default", "simplified", "simplified"),
        (
            np.array([
                [0.00175913, 0.01860749],
                [0.00556443, 0.02847149],
                [0.01163408, 0.04062758],
                [0.02087657, 0.05644548],
                [0.04660601, 0.09450612],
                [0.05526589, 0.10635019],
                [0.07993801, 0.13863351],
                [0.10629963, 0.17150553],
                [0.16364326, 0.23946384],
                [0.2978577, 0.38774834],
            ]),
            np.array([
                [0.00110793, 0.02924318],
                [0.00392459, 0.0399565],
                [0.00880861, 0.05311246],
                [0.01664273, 0.07008331],
                [0.0395029, 0.11036307],
                [0.04738362, 0.12277734],
                [0.07013454, 0.15640134],
                [0.09478413, 0.19038169],
                [0.1491697, 0.26002098],
                [0.27884435, 0.409966],
            ]),
            np.array([
                [-0.00144673, 0.01294098],
                [0.00201053, 0.02327682],
                [0.00792962, 0.03574854],
                [0.01711799, 0.05184753],
                [0.04292776, 0.09040558],
                [0.05163896, 0.10238403],
                [0.07648186, 0.13501239],
                [0.10304711, 0.16821726],
                [0.16086592, 0.23683523],
                [0.29625357, 0.38650505],
            ]),
            np.array([
                [-0.00479434, 0.0162886],
                [-0.00293753, 0.02822489],
                [0.00145694, 0.04222122],
                [0.00903741, 0.05992811],
                [0.03188101, 0.10145232],
                [0.03983201, 0.11419098],
                [0.06286346, 0.14863079],
                [0.08788386, 0.1833805],
                [0.14319001, 0.25451114],
                [0.27525461, 0.40750401],
            ]),
        ),
        strict=True,
    ),
)
def test_propn_ci_multinomial(
    _g_method: Literal["goodman", "quesenberry-hurst"],
    _g_alt: Literal["default", "simplified"],
    _t_val: NDArray[np.float64],
) -> None:
    assert_array_almost_equal(
        pci.propn_ci_multinomial(
            ques_data()["Frequency"], method=_g_method, alternative=_g_alt
        ),
        _t_val,
        decimal=6,
    )
