"""
Functions to estimate confidence intervals for
    (a.) a proportion or muliple proportions, and (b.) contrast between
    two independent proportions or two series of independent propotions.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NBitBase, NDArray
from scipy.optimize import OptimizeResult, root  # type: ignore
from scipy.stats import beta, chi2, norm  # type: ignore

from .. import VERSION  # noqa: TID252

__version__ = VERSION

TI = TypeVar("TI", bound=NBitBase)


def propn_ci(
    _npos: NDArray[np.integer[TI]] | int = 4,
    _nobs: NDArray[np.integer[TI]] | int = 10,
    /,
    *,
    alpha: float = 0.05,
    method: Literal[
        "Agresti-Coull", "Clopper-Pearson", "Exact", "Wilson", "Score"
    ] = "Wilson",
) -> tuple[
    NDArray[np.float64] | float,
    NDArray[np.float64] | float,
    NDArray[np.float64] | float,
    NDArray[np.float64] | float,
]:
    """Returns point estimates and confidence interval for a proportion

    Methods "Clopper-Pearson" and "Exact" are synoymous [3]_.  Similarly,
    "Wilson" and "Score" are synonyms here.

    Parameters
    ----------
    _npos
        Number of positives

    _nobs
        Number of observed values

    alpha
        Significance level

    method
        Method to use for estimating confidence interval

    Returns
    -------
        Raw and estimated proportions, and bounds of the confidence interval


    References
    ----------

    .. [3] Alan Agresti & Brent A. Coull (1998) Approximate is Better
       than “Exact” for Interval Estimation of Binomial Proportions,
       The American Statistician, 52:2, 119-126,
       https://doi.org/10.1080/00031305.1998.10480550

    """

    for _f in _npos, _nobs:
        if not isinstance(_f, int | np.integer):
            raise ValueError(
                f"Count, {_f!r} must have type that is a subtype of np.integer."
            )

    if not _nobs:
        return (np.nan, np.nan, np.nan, np.nan)

    _raw_phat: NDArray[np.float64] | float = _npos / _nobs
    _est_phat: NDArray[np.float64] | float
    _est_ci_l: NDArray[np.float64] | float
    _est_ci_u: NDArray[np.float64] | float

    match method:
        case "Clopper-Pearson" | "Exact":
            _est_ci_l, _est_ci_u = (
                beta.ppf(*_f)
                for _f in (
                    (alpha / 2, _npos, _nobs - _npos + 1),
                    (1 - alpha / 2, _npos + 1, _nobs - _npos),
                )
            )
            _est_phat = 1 / 2 * (_est_ci_l + _est_ci_u)

        case "Agresti-Coull":
            _zsc = norm.ppf(1 - alpha / 2)
            _zscsq = _zsc * _zsc
            _adjmt = 4 if alpha == 0.05 else _zscsq
            _est_phat = (_npos + _adjmt / 2) / (_nobs + _adjmt)
            _est_ci_l, _est_ci_u = (
                _est_phat + _g
                for _g in [
                    _f * _zsc * np.sqrt(_est_phat * (1 - _est_phat) / (_nobs + _adjmt))
                    for _f in (-1, 1)
                ]
            )

        case "Wilson" | "Score":
            _zsc = norm.ppf(1 - alpha / 2)
            _zscsq = _zsc * _zsc
            _est_phat = (_npos + _zscsq / 2) / (_nobs + _zscsq)
            _est_ci_l, _est_ci_u = (
                _est_phat
                + _f
                * _zsc
                * np.sqrt(_nobs * _raw_phat * (1 - _raw_phat) + _zscsq / 4)
                / (_nobs + _zscsq)
                for _f in (-1, 1)
            )

        case _:
            raise ValueError(f"Method, {f'"{method}"'} not yet implemented.")

    return _raw_phat, _est_phat, _est_ci_l, _est_ci_u


def propn_ci_multinomial(
    _counts: NDArray[np.integer[TI]],
    /,
    *,
    alpha: float = 0.05,
    method: Literal["goodman", "quesenberry-hurst"] = "goodman",
    alternative: Literal["default", "simplified"] = "default",
) -> NDArray[np.float64]:
    """Confidence intervals for multiple proportions.

    Parameters
    ----------
    _counts
        `n x 2` np.array of multinomial counts
    alpha
        Significance level
    method
        Method used to computer confidence intervals
    alternative
        Method used to estimate standard errors, whether "default"
        or "simplified"

    Returns
    -------
        Array of confidence intervals

    """
    if method not in (_mli := ("goodman", "quesenberry-hurst")):
        raise ValueError(
            f'Invalid value {f'"{method}"'} for "method". Must be one of {_mli}.'
        )

    _n = np.einsum("j->", _counts).astype(np.int64)
    _prob = _counts / _n
    _chi2_cr = (
        chi2(len(_counts) - 1).ppf(1 - alpha)
        if method == "quesenberry-hurst"
        else chi2(1).ppf(1 - (alpha / len(_counts)))
    )

    if alternative == "default":
        _ci_len_half = np.sqrt(_chi2_cr * (_chi2_cr + 4 * _n * _prob * (1 - _prob)))
        return np.column_stack([
            (_chi2_cr + 2 * _counts + _f * _ci_len_half) / (2 * (_n + _chi2_cr))
            for _f in (-1, 1)
        ])

    elif alternative == "simplified":
        _ci_len_half = np.sqrt(_chi2_cr * _prob * (1 - _prob) / _n)
        return np.column_stack([_prob + _f * _ci_len_half for _f in (-1, 1)])

    else:
        raise ValueError(
            f"Invalid value, {f'"{alternative}"'} for, \"alternative\". "
            f"Must be one of '{'("default", "simplified")'}'."
        )


def propn_diff_ci(
    _npos1: int = 4,
    _nobs1: int = 10,
    _npos2: int = 4,
    _nobs2: int = 10,
    /,
    *,
    alpha: float = 0.05,
    method: Literal["Agresti-Caffo", "Mee", "M-N", "Newcombe", "Score"] = "M-N",
) -> tuple[float, float, float, float]:
    R"""Confidence intervals for differences in binomial proportions.

    Methods available are Agresti-Caffo [4]_, Mee [5]_, Meitinen-Nurminen [5]_ [6]_
    and Newcombe (aka, Score method) [5]_. See also, source code for the
    R-language function BinomDiffCI, in the module StatsAndCIs [7]_.

    Parameters
    ----------
    _npos1, _npos2
        Counts of positive outcomes in the respective binomial distributions
    _nobs1, _nobs2
        Counts of all outcomes in the respective binomial distributions
    alpha
        Significance level
    method
        Method used to compute confidence intervals

    Returns
    -------
        Raw and expected values of estimated difference, with bounds of c.i.

    References
    ----------

    .. [4] Agresti, A., & Caffo, T. (2000). Simple and Effective
       Confidence Intervals for Proportions and Differences of Proportions
       Result from Adding Two Successes and Two Failures.
       The American Statistician, 54(4), 280--288. https://doi.org/10.2307/2685779

    .. [5] Newcombe, R.G. (1998). Two-sided confidence intervals for
       the single proportion: comparison of seven methods. Statist. Med., 17: 857-872.
       https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8%3C857::AID-SIM777%3E3.0.CO;2-E

    .. [6] Miettinen, O. and Nurminen, M. (1985). Comparative analysis of two rates.
        Statist. Med., 4: 213-226. https://doi.org/10.1002/sim.4780040211; Appendix I

    .. [7] StatsAndCIs.r, function BinomDiffCI, method, "mn"
       https://github.com/cran/DescTools/blob/master/R/StatsAndCIs.r
       (R source code is distributed under the CC-BY license.)

    """
    for _f in _npos1, _nobs1, _npos1, _nobs2:
        if not isinstance(_f, int | np.integer):
            raise ValueError(
                f"Count, {_f!r} must be of int type or be a subtype of np.integer."
            )

    if not min(_nobs1, _nobs2):
        return (np.nan, np.nan, np.nan, np.nan)

    match method:
        case "Agresti-Caffo":
            _res = _propn_diff_ci_agresti_caffo(
                _npos1, _nobs1, _npos2, _nobs2, alpha=alpha
            )

        case "Newcombe" | "Score":
            _res = _propn_diff_ci_newcombe_score(
                _npos1, _nobs1, _npos2, _nobs2, alpha=alpha
            )

        case "M-N" | "Mee":
            _res = _propn_diff_ci_mn(
                _npos1, _nobs1, _npos2, _nobs2, alpha=alpha, method=method
            )

        case _:
            raise ValueError(f"Method, {f'"{method}"'} not implemented.")

    return _res


def _propn_diff_ci_agresti_caffo(
    _npos1: int = 4,
    _nobs1: int = 10,
    _npos2: int = 4,
    _nobs2: int = 10,
    /,
    *,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """
    Estimate Agresti-Caffo confidence intervals for differences of
    multiple proportions.
    """

    _diff_hat = _npos1 / _nobs1 - _npos2 / _nobs2

    _zsc = norm.ppf(1 - alpha / 2)
    _zscsq = _zsc * _zsc

    _adjmt_t = 2 if alpha == 0.05 else _zscsq / 2
    _npos1_ac, _npos2_ac = (_f + _adjmt_t / 2 for _f in (_npos1, _npos2))
    _nobs1_ac, _nobs2_ac = (_f + _adjmt_t for _f in (_nobs1, _nobs2))

    _p1_est = _npos1_ac / _nobs1_ac
    _p2_est = _npos2_ac / _nobs2_ac
    _diff_est = _p1_est - _p2_est
    _se_est = np.sqrt(
        _p1_est * (1 - _p1_est) / _nobs1_ac + _p2_est * (1 - _p2_est) / _nobs2_ac
    )

    _diff_cl_l, _diff_cl_u = (_diff_est + _s * _zsc * _se_est for _s in (-1, 1))

    return _diff_hat, _diff_est, max(-1.0, _diff_cl_l), min(1.0, _diff_cl_u)


def _propn_diff_ci_newcombe_score(
    _npos1: int = 4,
    _nobs1: int = 10,
    _npos2: int = 4,
    _nobs2: int = 10,
    /,
    *,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """
    See Neccombe(1998), Agrest-Caffo (2002).
    """
    _l1, _u1 = propn_ci(_npos1, _nobs1, alpha=alpha, method="Wilson")[-2:]
    _l2, _u2 = propn_ci(_npos2, _nobs2, alpha=alpha, method="Wilson")[-2:]

    _zsc = norm.ppf(1 - alpha / 2)
    _diff_hat = _npos1 / _nobs1 - _npos2 / _nobs2

    _diff_cl_l = _diff_hat - _zsc * np.sqrt(
        _l1 * (1 - _l1) / _nobs1 + _u2 * (1 - _u2) / _nobs2
    )
    _diff_cl_u = _diff_hat + _zsc * np.sqrt(
        _u1 * (1 - _u1) / _nobs1 + _l2 * (1 - _l2) / _nobs2
    )

    return _diff_hat, (_diff_cl_l + _diff_cl_u) / 2, _diff_cl_l, _diff_cl_u


def _propn_diff_ci_mn(
    _npos1: int = 4,
    _nobs1: int = 10,
    _npos2: int = 4,
    _nobs2: int = 10,
    /,
    *,
    alpha: float = 0.05,
    method: Literal["M-N", "Mee"] = "M-N",
) -> tuple[float, float, float, float]:
    """
    See Miettinen and Nurminen (1985; Newcombe (1998);
        and StasAndCIs.r -> BinomDiffCi -> "mn".

    """
    for _f in _npos1, _nobs1, _npos1, _nobs2:
        if not isinstance(_f, int | np.integer):
            raise ValueError(
                f"Count, {_f!r} must have type that is a subtype of np.integer."
            )

    _chi_sq_cr = chi2.ppf(1 - alpha, 1)
    _counts = (_npos1, _nobs1, _npos2, _nobs2)

    _diff_hat = _npos1 / _nobs1 - _npos2 / _nobs2

    _ci_est_start = np.array([(_diff_hat + _s) / 2 for _s in (-1, 1)])
    # Avoid potential corner cases
    _ci_est_offset = (1 - 1.055e-2, 1)
    if _diff_hat == 1.0:
        _ci_est_start += _ci_est_offset
    elif _diff_hat == -1.0:
        _ci_est_start -= _ci_est_offset[::-1]

    def _obj_fn(
        _dh: float, _counts: Sequence[int], _cr: float, _method: Literal["M-N", "Mee"]
    ) -> float:
        return _cr - _propn_diff_chisq_mn(_counts, _dh, method=_method)

    def _get_sol(_sol: OptimizeResult, /) -> float:
        return float(_sol.x[0] if _sol.x.shape else _sol.x)

    _diff_cl_l, _diff_cl_u = (
        _get_sol(root(_obj_fn, _dh0, args=(_counts, _chi_sq_cr, method)))
        for _dh0 in _ci_est_start
    )

    _ci_lo, _ci_hi = max(-1.0, _diff_cl_l), min(1.0, _diff_cl_u)
    return _diff_hat, (_ci_lo + _ci_hi) / 2, _ci_lo, _ci_hi


def _propn_diff_chisq_mn(
    _counts: Sequence[int],
    _rd: float = 0.0,
    /,
    *,
    method: Literal["M-N", "Mee"] = "M-N",
) -> float:
    R"""Estimate the :math:`\chi^2` statistic for the Meittinen-Nurminen (1985),
    and Newcombe (1998) confidence intervals for a difference in binomial proportions.

    Parameters
    ----------
    _counts
        Numbers of positives and observations for (two) samples to be tested

    _rd
        Starting value

    method
        Specify Meitinen-Nurminen or Mee

    Returns
    -------
        Chi-square estimate

    """
    if _counts is None:
        _counts = [1] * 4

    _np1, _no1, _np2, _no2 = _counts
    _p1h, _p2h = _np1 / _no1, _np2 / _no2
    _diff = _p1h - _p2h - _rd

    if not _diff:
        return 0.0

    _np, _no = _np1 + _np2, _no1 + _no2

    _l3 = _no
    _l2 = (_no1 + 2 * _no2) * _rd - _no - _np
    _l1 = (_no2 * _rd - _no - 2 * _np2) * _rd + _np
    _l0 = _np2 * _rd * (1 - _rd)
    _l2_to_3l3 = _l2 / (3 * _l3)

    _q = _l2_to_3l3**3 - (_l1 * _l2_to_3l3 - _l0) / (2 * _l3)
    _p = np.sign(_q) * np.sqrt(_l2**2 - 3 * _l3 * _l1) / (3 * _l3)
    _a = (np.pi + np.arccos(_q / _p**3)) / 3

    _p2t: float = 2 * _p * np.cos(_a) - _l2_to_3l3
    _p1t: float = _p2t + _rd

    return _diff**2 / (
        (_p1t * (1 - _p1t) / _no1 + _p2t * (1 - _p2t) / _no2)
        * (_no / (_no - 1) if method == "M-N" else 1.0)
    )


def propn_diff_ci_multinomial(
    _counts: NDArray[np.integer[TI]], /, *, alpha: float = 0.05
) -> NDArray[np.float64]:
    """Estimate confidence intervals of pair-wise differences in multinomial proportions

    Differences in multinomial proportions sum to zero.

    Parameters
    ----------
    _counts
        Two dimensional np.array of observed values of multinomial distributions
        (in columns).
    alpha
        Significance level

    Returns
    -------
        Array of confidence intervals

    """

    if len(_counts.shape) > 2:
        raise ValueError(
            "This implementation is only valid for estimating confidence intervals "
            "for differences in two (2) sets of multinomial proportions."
        )

    _prob = _counts / np.einsum("jk->k", _counts).astype(np.int64)
    _var = np.einsum("jk->j", _prob * (1 - _prob) / _counts)[:, None]

    _d, _d_cr = np.diff(_prob, axis=1), norm.ppf(1 - (alpha / len(_counts)))
    return np.column_stack([_d + _f * _d_cr * np.sqrt(_var) for _f in (-1, 1)])


@dataclass(slots=True, frozen=True)
class MultinomialPropnsTest:
    estimate: np.float64
    dof: int
    critical_value: np.float64
    p_value: np.float64


def propn_test_multinomial(
    _counts: NDArray[np.integer[TI]], /, *, alpha: float = 0.05
) -> MultinomialPropnsTest:
    """Chi-square test for homogeneity of differences in multinomial proportions.

    Differences in multinomial proportions sum to zero.

    Parameters
    ----------
    _counts
        Two dimensional array of observed values of multinomial distributions
        (in columns).
    alpha
        Significance level

    Returns
    -------
        Estimated statistic, degrees of freedom, critical value, p-value

    """

    _n = np.einsum("jk->", _counts).astype(np.int64)
    _n_k = np.einsum("jk->k", _counts).astype(np.int64)
    _prob = _counts / _n_k

    _p_bar = _n / np.einsum("jk->j", _n_k / _prob)

    _y_sq = _n * ((1 / np.einsum("j->", _p_bar)) - 1)
    _dof = np.array([_s - 1 for _s in _counts.shape]).prod()
    _chi_rv = chi2(_dof)

    return MultinomialPropnsTest(
        _y_sq, _dof, _chi_rv.ppf(1 - alpha), 1 - _chi_rv.cdf(_y_sq)
    )
