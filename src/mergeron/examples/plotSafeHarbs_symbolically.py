"""

    Solve for, and plot, weighted-average GUPPI boundaries using sympy,
    as a check against the numerical solutions in mergeron.core.guidelines_standards

"""

from collections.abc import Sequence

from sympy import solve, symbols
from sympy.plotting import plot as symplot


def plot_safeharb_boundaries(_g_bar: float, _r_bar: float, _m_vals: Sequence):
    s1, s2, rbar, delta_star = symbols(r"s1 s2 \overline{r} \delta^*", real=True)

    for _m_val in _m_vals:
        s_m = round((dstar_val := _g_bar / (_r_bar * _m_val)) / (1 + dstar_val), 3)

        # Own-share-weighted average GUPPI, "inside-out" recapture
        s2_sol = solve(
            s1 * s2 / (1 - s1)
            + s2 * s1 / (1 - s2 * _r_bar - s1 * (1 - _r_bar))
            - (s1 + s2) * delta_star,
            s2,
        )
        symplot(
            s2_sol[0].subs({delta_star: dstar_val}),
            (s1, 0, s_m),
            ylabel=s2,
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            axis_center=(0.0, 0.0),
        )

        # Cross-product-share-weighted average GUPPI, proportional recapture
        s2_sol = solve(
            s2 * s2 / (1 - s1) + s1 * s1 / (1 - s2) - delta_star * (s1 + s2), s2
        )
        symplot(
            s2_sol[1].subs({delta_star: dstar_val}),
            (s1, 0, s_m),
            ylabel=s2,
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            axis_center=(0.0, 0.0),
        )


if __name__ == "__main__":
    plot_safeharb_boundaries(0.06, 0.80, (1.00, 0.67, 0.30))
