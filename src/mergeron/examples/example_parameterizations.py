"""

    A few parameters/relations employed in policy analysis using this pacakage
    are demonstrated here

"""

from typing import NamedTuple

from numpy import column_stack, divide
from numpy.typing import NDArray

from mergeron.core.psuedorandom_numbers import prng
from mergeron.gen.data_generation import (
    RECConstants,
    gen_market_shares_dirichlet,
    gen_market_shares_uniform,
)


def gen_rval_mnl(
    _ssz: int = 10**8, _r_bar: float = 0.80, *, pcm_dist_type: str = "Uniform"
) -> NamedTuple:
    """

    :param _ssz: sample size
    :param _r_bar: recapture rate
    :param pcm_dist_type: margin distribution name
    :return:
    """

    # Define output type
    class RvalMNL(NamedTuple):
        qtyshr_array: NDArray[float]
        pcm_array: NDArray[float]
        mnl_test_rows: NDArray[int]

    _qtyshr_array = gen_market_shares_uniform(_ssz).mktshr_array[:, :2]

    _qtyshr_min = _qtyshr_array.min(axis=1, keepdims=True, initial=None)
    _cprob = divide(_r_bar, 1 - (1 - _r_bar) * _qtyshr_min)
    del _qtyshr_min

    _purchprob_array = _cprob * _qtyshr_array

    _pcm0: NDArray[float]
    if pcm_dist_type == "Uniform":
        _pcm0 = prng().uniform(size=(_ssz, 1))
    elif pcm_dist_type == "Beta":
        _pcm0 = prng().beta(10, 10, size=(_ssz, 1))
    else:
        raise ValueError("Invalid type for distribution of margins")

    _pcm1 = divide(
        _pcm0 * (1 - _purchprob_array[:, [0]]), (1 - _purchprob_array[:, [1]])
    )

    _mnl_test_rows = _pcm1.__gt__(0) & _pcm1.__lt__(1)
    _pcm_array = column_stack((_pcm0, _pcm1))

    return RvalMNL(_qtyshr_array, _pcm_array, _mnl_test_rows)


def gen_rval_ssp(_ssz: int = 10**6, _r_bar: float = 0.8) -> NamedTuple:
    R"""Given r_1 (_r_bar), generates r_2 and diversion ratios under share proportionality

    More testing with::

        from sympy import solve, symbols
        s1, s2, m1, mstar, rbar, dstar = symbols('s_1, s_2, m_1, m^{*}, \overline{r}, \delta^{*}')
        infl2_pts = solve(
            (
                s2 - dstar + dstar * rbar * s1 + dstar * (1 - rbar) * s2,
                s1 - (1 - s2) * dstar * mstar / m1
            ),
            s1, s2)

    Substitute various values into the solution to test against plots generated elsewhere
    """

    # Define output type
    class RvalSSP(NamedTuple):
        """Container for share, recapture ratio, and diversion rate arrays"""

        qtyshr_array: NDArray[float]
        r_val: NDArray[float]
        divr_array: NDArray[float]

    _qtyshr_array = gen_market_shares_uniform(_ssz).mktshr_array[:, :2]

    _qtyshr_min = _qtyshr_array.min(axis=1, keepdims=True, initial=None)
    _cprob = divide(_r_bar, 1 - (1 - _r_bar) * _qtyshr_min)
    del _qtyshr_min

    _r_val = divide(_cprob * (1 - _qtyshr_array), 1 - _cprob * _qtyshr_array)

    _divr_array = _r_val * _qtyshr_array[:, ::-1] / (1 - _qtyshr_array)

    return RvalSSP(_qtyshr_array, _r_val, _divr_array)


def gen_implied_mkt_shr_1(_fcount: int = 5) -> NDArray[float]:
    """
    Generate implied market shares for firm 1 with all pairs of products
    in a putative market of `_fcount` firms.

    Parameters
    ----------
    _fcount: Firm count for the given market

    Returns
    -------
    Vector of implied market shares

    """

    _mkt_sample = gen_market_shares_dirichlet([1] * _fcount, 10, RECConstants.OUTIN)

    _mktshr_array = _mkt_sample.mktshr_array
    _chprob_outside_good = _mkt_sample.choice_prob_outgd

    _recapture_array = divide(
        (1 - _chprob_outside_good) * (1 - _mktshr_array),
        1 - (1 - _chprob_outside_good) * _mktshr_array,
    )

    _divratio_1j = divide(
        _recapture_array[:, :1] * _mktshr_array[:, 1:], 1 - _mktshr_array[:, :1]
    )
    _divratio_j1 = divide(
        _recapture_array[:, 1:] * _mktshr_array[:, :1], 1 - _mktshr_array[:, 1:]
    )

    _implied_mkt_shr_1 = divide(
        _divratio_j1 * (_recapture_array[:, :1] - _divratio_1j),
        _recapture_array[:, :1] * _recapture_array[:, 1:] - _divratio_1j * _divratio_j1,
    )

    return _implied_mkt_shr_1


if __name__ == "__main__":
    print(gen_implied_mkt_shr_1(10))
