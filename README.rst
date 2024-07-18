mergeron: Merger Policy Analysis using Python
=============================================

Download and analyze merger investigations data published by the U.S. Federal Trade Commission in various reports on extended merger investigations during 1996 to 2011. Model the sets of mergers conforming to various U.S. Horizontal Merger Guidelines standards. Analyze intrinsic clearance rates and intrinsic enforcement rates under Guidelines standards using generated data with specified distributions of market shares, price-cost margins, firm counts, and prices, optionally imposing restrictions impled by statutory filing thresholds and/or Bertrand-Nash oligopoly with MNL demand.

Intrinsic clearance and enforcement rates are distinguished from *observed* clearance and enforcement rates in that the former do not reflect the effects of screening and deterrence as do the latter.


Introduction
------------

Classes for specifying concentration standards (`mergeron.core.guidelines_boundaries.ConcentrationBoundary`) and diversion-ratio standards (`mergeron.core.guidelines_boundaries.DiversionRatioBoundary`), with automatic generation of boundary (as an array of share-pairs) and area, are provided in `mergeron.core.guidelines_boundaries`. This module also includes a function for generating plots of concentation and diversion-ratio boundaries, and functions for mapping GUPPI standards to concentration (ΔHHI) standards, and vice-versa.

Methods for generating industry data under various distributions of shares, margins, and prices are included in, `mergeron.gen.data_generation`. Shares are drawn with uniform distribution with :math:`s_1 + s_2 \leqslant 1` and an unspecified number of firms. Alternatively, shares may be drawn from the Dirichlet distribution. When drawing shares from the Dirichlet distribution, the user can specify a fixed number for firms or provide a vector of weights specifying the frequency distribution over sequential firm counts, e.g., :code:`[133, 184, 134, 52, 32, 10, 12, 4, 3]` to specify shares drawn from Dirichlet distributions with 2 to 10 pre-merger firms distributed as in data for FTC merger investigations during 1996--2003 (See, for example, Table 4.1 of `FTC, Horizontal Merger Investigations Data, Fiscal Years 1996--2003 (Revised: August 31, 2004) <"https://www.ftc.gov/sites/default/files/documents/reports/horizontal-merger-investigation-data-fiscal-years-1996-2003/040831horizmergersdata96-03.pdf>`_). The user can specify recapture rates as, "proportional", "inside-out" --- i.e., consistent with merging-firms' in-market shares and a default recapture rate) --- or "outside-in" --- i.e., purchase probabilities are drawn at random for :math:`N+1` goods, from which are derived market shares and recapture rates for the :math:`N` goods in the putative market. Documentation on specifying the sampling strategy for market shares is at `mergeron.gen.ShareSpec`. Price-cost-margins may be specified as symmetric, i.i.d., or subject to equilibrium conditions for (profit-mazimization in) Bertrand-Nash oligopoly with MNL demand (see, `mergeron.gen.PCMSpec`). Prices may be specified as symmetric or asymmetric, and in the latter case, the direction of correlation between merging firm prices, if any, can also be specified (see, `mergeron.gen.PriceSpec`). Two alternative approaches for modeling statutory filing requirements (HSR filing thresholds) are implemented (see, `mergeron.gen.SSZConstants`). The full specification of a market sample is given in a `mergeron.gen.market_sample.MarketSample` object. Data are drawn by invoking `mergeron.gen.market_sample.MarketSample.generate_sample` which adds a `data` property of class, `mergeron.gen.MarketDataSample`. Enforcement or clearance counts are computed by invoking `mergeron.gen.market_sample.MarketSample.estimate_invres_counts`, which adds an `invres_counts` property of class `mergeron.gen.UPPTestsCounts`. For fast, parallel generation of enforcement or clearance counts over large market data samples that ordinarily would exceed available limits on machine memory, the user can invoke the method `estimate_invres_counts` on a `mergeron.gen.market_sample.MarketSample` object without first invoking `generate_sample`. Note, however, that this strategy discards the market sample in the interests of conserving memory and maintaining high performance.

Methods for printing enforcement statistics based on FTC investigations data and test data are printed to screen or rendered to LaTex files (for processing into publication-quality tables) using methods provided in `mergeron.gen.enforcement_stats`.

Programs demonstrating the analysis and reporting facilites provided by the sub-package, `mergeron.demo`.

This package exposes methods employed for generating random numbers with selected continuous distribution over specified parameters, and with CPU multithreading on machines with multiple virtual, logical, or physical CPU cores. To access these directly:

.. code-block:: python

    import mergeron.core.pseudorandom_numbers as prng

Also included are methods for estimating confidence intervals for proportions and for contrasts (differences) in proportions. (Although coded from scratch using the source literature, the APIs implemented in the module included here are designed for consistency with the APIs in, `statsmodels.stats.proportion` from the package, `statsmodels` (https://pypi.org/project/statsmodels/).) To access these directly:

.. code-block:: python

    import mergeron.core.proportions_tests as prci

A recent version of Paul Tol's python module, `tol_colors.py` is redistributed within this package. Other than re-formatting and type annotation, the `mergeron.ext.tol_colors` module is re-distributed as downloaded from, https://personal.sron.nl/~pault/data/tol_colors.py. The `tol_colors.py` module is distributed under the Standard 3-clause BSD license. To access the `mergeron.ext.tol_colors` module directly:

.. code-block:: python

    import mergeron.ext.tol_colors as ptc

Documentation for this package is in the form of the API Reference. Documentation for individual functions and classes is accessible within a python shell. For example:

.. code-block:: python

    import mergeron.core.market_sample as market_sample

    help(market_sample.MarketSample)


.. image:: https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json
   :alt: Poetry
   :target: https://python-poetry.org/

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :alt: Ruff
   :target: https://github.com/astral-sh/ruff

.. image:: https://www.mypy-lang.org/static/mypy_badge.svg
   :alt: Checked with mypy
   :target: https://mypy-lang.org/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :alt: License: MIT
   :target: https://opensource.org/licenses/MIT
