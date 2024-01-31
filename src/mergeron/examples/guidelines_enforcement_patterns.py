from __future__ import annotations

from numpy import array, column_stack

import mergeron.core.proportions_tests as pcl

enf_counts = array((
    (24, 25, 33, 33),
    (30, 33, 27, 32),
    (14, 15, 9, 11),
    (13, 22, 6, 11),
    (6, 8, 7, 9),
    (1, 3, 2, 4),
    (1, 3, 2, 4),
    (1e-15, 1e-15, 1, 1),
))

if __name__ == "__main__":
    adj_n = len(enf_counts)

    print()
    print("C.I.s for relative frequency of II Requests")
    cis_1996_2003, cis_2004_2011 = (
        pcl.propn_ci_multinomial(_c) for _c in (enf_counts[:, _i] for _i in (1, 3))
    )
    for cdx, ci in enumerate(cis_1996_2003):
        print(ci, ";", cis_2004_2011[cdx, :])

    print()
    print(
        "C.I.s for relative frequency of II Requests, 1996-2003",
        "(Bonnferoni-adjusted Wilson C.I.s)",
    )
    for count in (ic_1996_2003 := enf_counts[:, 1]):
        print(
            pcl.propn_ci(
                count, ic_1996_2003.sum(), alpha=0.05 / adj_n, method="Wilson"
            )[2:]
        )
    del count

    print("Conf. intervals for relative frequency of II Requests (Goodman, 1965)")
    for goodman_alternative in "default", "simplified":
        for goodman_method in "goodman", "quesenberry-hurst":
            print(f"Method, {goodman_method!r}; alternative, {goodman_alternative!r}")
            for _cis in column_stack([
                pcl.propn_ci_multinomial(
                    enf_counts[:, _cidx],
                    method=goodman_method,
                    alternative=goodman_alternative,
                )
                for _cidx in (1, 3)
            ]):
                print(_cis)
            print()

    print()
    print("Conf. intervals for differences in proportions enforced (Goodman, 1964)")
    print(repr(pcl.propn_ci_diff_multinomial(enf_counts[:, [1, 3]])))

    print()
    print("Goodman's chi-squared test for homogeneity of enforcement patterns")
    print(repr(pcl.propn_diff_multinomial_chisq(enf_counts[:, [1, 3]])))

    print()
    print(
        "C.I.s for differences in proportions enforced",
        "(Bonnferoni-adjusted Newcombe C.I.s)",
    )
    for counts in enf_counts:
        print(pcl.propn_diff_ci(*counts, alpha=0.05 / adj_n, method="Newcombe")[2:])
    print()
