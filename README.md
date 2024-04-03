mergeron: Merger Policy Analysis with Python
============================================

Download and analyze merger investigations data published by the U.S. Federal Trade Commission in various reports on extended merger investigations during 1996 to 2011. Model the sets of mergers conforming to various U.S. Horizontal Merger Guidelines standards. Analyze intrinsic clearance rates and intrinsic enforcement rates under Guidelines standards using generated data with specified distributions of market shares, price-cost margins, firm counts, and prices, optionally imposing restrictions impled by statutory filing thresholds and/or Bertrand-Nash oligopoly with MNL demand.

Intrinsic clearance and enforcement rates are distinguished from *observed* clearance and enforcement rates in that the former do not reflect the effects of screening and deterrence as do the latter.


Modules of primary interest
---------------------------

Methods for plotting boundaries of (sets of mergers falling within) specified concentration and diversion-ratio boundaries are in `mergeron.core.guidelines_boundaries`. This module also includes functions for calibrating GUPPI thresholds to concentration (ΔHHI) thresholds, and vice-versa.

Methods for generating industry data under various distributions of shares, prices, and margins are included in, `mergeron.gen.data_generation`. The user can specify whether rates are specified as, "proportional", "inside-out" (i.e., consistent with merging-firms' in-market shares and a default recapture rate), or "outside-in", (i.e., purchase probabilities are drawn at random for :math:`n+1` goods, from which are derived market shares and recapture rates for the :math:`n` goods in the putative market). Price-cost-margins may be specified as symmetric, i.i.d, or consistent with equilibrium conditions for (profit-mazimization in) Bertrand-Nash oligopoly with MNL demand. Prices may be specified as symmetric or asymmetric, and in the latter case, the direction of correlation between merging firm prices, if any, can also be specified. Two alternative approaches for modeling statutory filing requirements (HSR filing thresholds) are implemented.

Methods for testing generated industry data against criteria for gross upward pricing pressure ("GUPPI") with and without a diversion ratio limit, critical marginal cost reduction ("CMCR"), and indicative price rise ("IPR")/partial merger simulation are included in the module, `mergeron.gen.guidelines_tests`. Test data are constructed in parallel and the user can specify the number of `threads` and sub-sample size for each thread to manage CPU and memory utilization.

FTC investigations data and test data are printed to screen or rendered to LaTex files (for processing into publication-quality tables) using methods provided in `mergeron.gen.investigations_stats`.

Programs demonstrating the analysis and reporting facilites provided by the sub-package, `mergeron.examples`.

This package exposes methods employed for generating random numbers with selected continuous distribution over specified parameters, and with CPU multithreading on machines with multiple virtual, logical, or physical CPU cores. To access these directly:

    import mergeron.core.pseudorandom_numbers as prng

Also included are methods for estimating confidence intervals for proportions and for contrasts (differences) in proportions. (Although coded from scratch using the source literature, the APIs implemented in the module included here are designed for consistency with the APIs in, `statsmodels.stats.proportion` from the package, `statsmodels` (https://pypi.org/project/statsmodels/).) To access these directly:

    import mergeron.core.proportions_tests as prci

A recent version of Paul Tol's python module, `tol_colors.py` is redistributed within this package. Other than re-formatting and type annotation, the `tol_colors` module is re-distributed as downloaded from, https://personal.sron.nl/~pault/data/tol_colors.py. The tol_colors.py module is distributed under the Standard 3-clause BSD license. To access the tol_colors module directly:

    import mergeron.ext.tol_colors

Documentation for this package is in the form of the API Reference. Documentation for individual functions and classes is also accessible within a python shell. For example:

    import mergeron.core.data_generation as dgl

    help(dgl.gen_market_sample)


[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
