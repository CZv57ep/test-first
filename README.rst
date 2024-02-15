mergeron: Merger Policy Analysis with Python
============================================

Download and analyze merger investigations data published by the U.S. Federal Trade Commission in various reports on extended merger investigations during 1996 to 2011. Model the sets of mergers conforming to various U.S. Horizontal Merger Guidelines standards. Analyze intrinsic clearance rates and intrinsic enforcement rates under Guidelines standards using generated data with specified distributions of market shares, price-cost margins, firm counts, and prices, optionally imposing restrictions impled by statutory filing thresholds and/or Bertrand-Nash oligopoly with MNL demand.

Intrinsic clearance and enforcement rates are distinguished from *observed* clearance and enforcement rates in that the former do not reflect the effects of screening and deterrence as do the latter.

Modules of primary interest
---------------------------

Routines for downloading and organizing FTC merger investigtions data published in 2004, 2007, 2008, and 2013. Includes a routine for constructing investigations data for non-overlapping periods, and other partitions on the data, subject to the constraints of the reported data.

    :code:`mergeron.core.ftc_merger_investigations_data`

Routines for plotting boundaries of (sets of mergers falling within) specified concentration and diversion ratio boundaries and for calibrating GUPPI thresholds to concentration (ΔHHI) thresholds, and vice-versa.

    :code:`mergeron.core.guidelines_standards`

Routines for generating industry data under various distributions of shares, prices, and margins. The user can specify whether rates are specified as, "proportional", "inside-out" (i.e., consistent with merging-firms' in-market shares and a default recapture rate), or "outside-in", (i.e., purchase probabilities are drawn at random for :math:`n+1` goods, from which are derived market shares and recapture rates for the :math:`n` goods in the putative market). Price-cost-margins may be specified as symmetric, i.i.d, or consistent with equilibrium conditions for (profit-mazimization in) Bertrand-Nash oligopoly with MNL demand. Prices may be specified as symmetric or asymmetric, and in the latter case, the direction of correlation between merging firm prices, if any, can also be specified. Two alternative approaches for modeling statutory filing requirements (HSR filing thresholds) are implemented.

    :code:`mergeron.gen.data_generation`

Routines for testing generated industry data against criteria on diversion ratio, gross upward pricing pressure ("GUPPI"), critical marginal cost reduction ("CMCR"), and indicative price rise ("IPR")/partial merger simulation. Test data are constructed in parallel and the user can specify number of `threads` and sub-sample size for each thread to manage CPU and memory utilization.

    :code:`mergeron.gen.guidelines_tests`

FTC investigations data and test data are printed to screen or rendered in LaTex to text files (for processing into publication-quality tables) using routines provided in :code:`mergeron.gen.invstigations_stats`.

Programs demonstrating the analysis and reporting facilites provided by the package.

    :code:`mergeron.examples`

This package also exposes routines employed for generating random numbers with selected continuous distribution over specified parameters, and with CPU multithreading on machines with multiple virtual, logical, or physical CPU cores. To access these directly:

    :code:`import mergeron.core.pseudorandom_numbers as prng`

Also included are routines for estimating confidence intervals for proportions and for contrasts (differences) in proportions. (These are a subset of routines available from the package, :code:`statsmodels` (https://pypi.org/project/statsmodels/), in the module, :code:`statsmodels.stats.proportion`, with selective modifications. To access these directly:

    :code:`import mergeron.core.proportions_tests as prci`

A recent version of Paul Tol's python module, :code:`tol_colors.py` is redistributed within this package. Other than re-formatting and type annotation, the :code:`tol_colors` module is re-distributed as downloaded from, https://personal.sron.nl/~pault/data/tol_colors.py. The tol_colors.py module is distributed under the Standard 3-clause BSD license. To access the tol_colors module directly:

    :code:`import mergeron.ext.tol_colors`

Documentation for this package is in the form of the API Reference. For individual functions and classes, this same documentation is accessible within a python shell. For example, the below snippet produces the API reference for the given function in a python::
:code:

    import mergeron.core.data_generation as dgl

    help(dgl.gen_market_sample)




