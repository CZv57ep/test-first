mergeron: Merger Policy Analysis with Python
============================================

Download and analyze merger investigations data published by the U.S. Federal Trade Commission in various reports on extended merger investigations during 1996 to 2011. Model the sets of mergers conforming to various U.S. Horizontal Merger Guidelines standards. Analyze intrinsic clearance rates and intrinsic enforcement rates under Guidelines standards generated data for specified distributions of market shares, price-cost margins, firm counts, and prices, optionally imposing restrictions implied by profit maximization and/or statutory filing thresholds.

Intrinsic clearance and enforcement rates are distinguished from *observed* clearance and enforcement rates in that the former do not reflect the effects of screening and deterrence.

Modules of primary interest
---------------------------

:code:`mergeron.core.ftc_merger_investigations_data`
    Routines for downloading and organizing FTC merger investigtions data published in 2004, 2007, 2008, and 2013. Includes a routine for constructing investigations data for non-overlapping periods, and other partitions on the data, subject to the constraints of the reported data.

:code:`mergeron.core.guidelines_standards`
    Routines for plotting boundaries of (sets of mergers falling within) specified concentration and diversion ratio boundaries and for calibrating GUPPI thresholds to concentration (ΔHHI) thresholds, and vice-versa.

:code:`mergeron.gen.data_generation`
    Routines for generating industry data under various distributions of shares, prices, and margins, with and without equilibrium restrictions from MNL demand. The user can specify whether recapture ratios are specified as, "proportional", "inside-out" (i.e., consistent with merging-firms' in-market shares and a default), or "outside-in", (i.e., purchase probabilities are drawn at random for `n+1` goods, from which are derived market shares for the `n` goods in the putative market). Price-cost-margins may be specified as symmetric, i.i.d, or consistent with Bertrand-Nash oligopoly with MNL demand. Prices may be specified as symmetric or asymmetric, and in the latter case, the direction of correlation between merging firm prices, if any, can also be specified. Two alternative approaches for modeling statutory filing requirements (HSR filing thresholds) are implemented.

:code:`mergeron.gen.guidelines_tests`
    Routines for testing generated industry data against criteria on diversion ratio, gross upward pricing pressure ("GUPPI"), critical marginal cost reduction ("CMCR"), and indicative price rise ("IPR")/partial merger simulation. Test data are constructed in parallel and the user can specify number of `threads` and sub-sample size for each thread to manage CPU and memory utilization.

FTC investigations data and test data are printed to screen or to publication-quality tables formatted in Latex using routines provided in :code:`mergeron.gen.invstigations_stats`.

:code:`mergeron.examples`
    Programs demonstrating the analysis and reporting facilites provided by the package.

This package also exposes routines employed for generating random numbers with selected continuous distribution over specified parameters, and with CPU multithreading on machines with multiple virtual, logical, or physical CPU cores. To access these directly:

:code:`import mergeron.core.pseudorandom_numbers as prng`

Also included are routines for estimating confidence intervals for proportions and for contrasts (differences) in proportions. (These are a subset of routines available from the package, :code:`statsmodels` (https://pypi.org/project/statsmodels/), in the module, :code:`statsmodels.stats.proportion`, with selective modifications. To access these directly:

:code:`import mergeron.core.proportions_tests as prci`

A recent version of Paul Tol's python module, :code:`tol_colors.py` is redistributed within this package. Other than re-formatting and type annotation, the :code:`tol_colors` module is re-distributed as downloaded from, https://personal.sron.nl/~pault/data/tol_colors.py. The tol_colors.py module is distributed under the Standard 3-clause BSD license. To access the tol_colors module directly:

:code:`import mergeron.ext.tol_colors`
