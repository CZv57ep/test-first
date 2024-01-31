mergeron: Merger Policy Analysis with Python
============================================

Model the standards specified in (or proposed for) U.S. Horizontal Merger Guidelines in terms of the sets of mergers conforming to a given Guidelines standard. Analyze intrinsic clearance rates and intrinsic enforcement rates under Guidelines standards, i.e., rates derived from applying a theoretically-derived model to generated data for specified distributions of relevant economic factors. Download and analyze merger investigations data published by FTC, covering investigations during the years, 1996 to 2011.

Intrinsic clearance and enforcement rates are distinct from observed clearance and enforcement rates in that the former do not reflect the effects of screening and deterrence.

Programs for replicating results in various of my papers are included in the "examples" sub-package.

This package also exposes functions for generating random numbers with selected continuous distribution over specified parameters, employing multithreading on the CPU. To access these directly:

    import mergeron.core.prng_cpu_multi

Also included are functions for estimating confidence intervals for proportions and contrasts in proportions; the Bonfferoni adjustment is applied for confidence intervals for multiple comparisons. To access these directly:

    import mergeron.core.propn_ci_lib

A recent version of Paul Tol's python module, "tol_colors.py" is included in this package. Other than re-formatting (black and isort), the tol_colors module, licensed under the Standard 3-clause BSD license, is re-distributed here as downloaded from, https://personal.sron.nl/~pault/data/tol_colors.py. To access the tol_colors module directly:

    import mergeron.ext.tol_colors
