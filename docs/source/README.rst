mergeron: Merger Policy Analysis using Python
=============================================

Analyze the sets of mergers conforming to concentration and diversion ratio bounds. Analyze intrinsic enforcement rates, and intrinsic clearance rates, under concentration, diversion ratio, GUPPI, CMCR, and IPR bounds using generated data with specified distributions of market shares, price-cost margins, firm counts, and prices, optionally imposing restrictions implied by statutory filing thresholds and/or Bertrand-Nash oligopoly with MNL demand. Download and analyze merger investigations data published by the U.S. Federal Trade Commission in various reports on extended merger investigations (Second Requests) during 1996 to 2011.

Here, enforcement rates derived with merger enforcement as being exogenous to firm conduct are defined as intrinsic enforcement rates, and similarly intrinsic clearance rates. Depending on the merger enforcement regime, or merger control regime, intrinsic enforcement rates may also not be the complement of intrinsic clearance rates, i.e, it is not necessarily true that the intrinsic clearance rate estimate for a given enforcement regime is 1 minus the intrinsic enforcement rate. In contrast, observed enforcement rates reflect the deterrent effects of merger enforcement on firm conduct as well as the effects of merger screening on the level of enforcement; and, by definition, the observed clearance rate is 1 minus the observed enforcement rate.

Introduction
------------

Module :mod:`mergeron.core.guidelines_boundaries` includes classes for specifying concentration bounds (:class:`mergeron.core.guidelines_boundaries.ConcentrationBoundary`) and diversion-ratio bounds (:class:`mergeron.core.guidelines_boundaries.DiversionRatioBoundary`), with automatic generation of boundary (as an array of share-pairs) and area. This module also includes a function for generating plots of concentration and diversion-ratio boundaries, and functions for mapping GUPPI standards to concentration (ΔHHI) standards, and vice-versa.

Module :mod:`mergeron.gen.market_sample` includes the :class:`mergeron.gen.market_sample.MarketSample` with methods for, (i) generating sample data under a rich specification of shares, diversion ratios, margins, prices, and HSR filing requirements, and (ii) for estimating enforcement or clearance rates under specified enforcement regimes given a method of aggregating diversion ratio or GUPPI estimates for the firms in a merger. Notably. share are generated not just for markets with a fixed number of firms, but for markets with multiple firm-count weights, which may be left unspecified or explicitly specified.

Unless otherwise specified, merging-firm shares are drawn with uniform distribution over the space :math:`s_1 + s_2 \leqslant 1` for an unspecified number of firms. Alternatively, shares may be drawn from the Dirichlet distribution, with specified shape parameters (see :class:`mergeron.gen.ShareConstants`. When drawing shares from the Dirichlet distribution, the user passes, using :attr:`mergeron.gen.MarketSpec.ShareSpec.firm_count_weights`, a vector of weights specifying the frequency distribution over sequential firm counts, e.g., :code:`[133, 184, 134, 52, 32, 10, 12, 4, 3]` to specify shares drawn from Dirichlet distributions with 2 to 10 pre-merger firms distributed as in data for FTC merger investigations during 1996--2003 (See, for example, Table 4.1 of `FTC, Horizontal Merger Investigations Data, Fiscal Years 1996--2003 (Revised: August 31, 2004) <https://www.ftc.gov/sites/default/files/documents/reports/horizontal-merger-investigation-data-fiscal-years-1996-2003/040831horizmergersdata96-03.pdf>`_). If :code:`mergeron.gen.MarketSpec.ShareSpec.firm_count_weights` is not assigned a value when defining :attr:`mergeron.gen.MarketSpec.ShareSpec` (which has type, :class:`mergeron.gen.ShareSpec`), the default values is used, with results in a sample of markets with 2 to 6 firms with equal relative frequency.

Recapture rates can be specified as, "proportional", "inside-out", "outside-in" (see :class:`mergeron.RECConstants`. The "inside-out" specification results in recapture ratios consistent with merging-firms' in-market shares and a default recapture rate. The "outside-in" specification yields diversion ratios from purchase probabilities drawn at random for :math:`N+1` goods, from which are derived market shares and recapture rates for the :math:`N` goods in the putative market (see, :class:`mergeron.gen.DiversionRatioSpec`). The "outside-in" specification is invalid when the distribution of markets over firm-count is unspecified, i.e., when :code:`mergeron.gen.MarketSpec.ShareSpec.dist_type ==`:attr:`mergeron.gen.ShareConstants.UNI`.

Price-cost-margins may be specified as having uniform distribution, Beta distribution (including a bounded Beta distribution with specified mean and variance), or an empirical distribution. The empirical margin distribution is based on resampling margin data published by Prof. Damodaran of NYU Stern School of Business (see Notes), using an estimated Gaussian KDE. The second merging firm's margin may be specified as symmetric, i.i.d., or subject to equilibrium conditions for (profit-maximization in) Bertrand-Nash oligopoly with MNL demand (see, :class:`mergeron.gen.PCMSpec`).

Prices may be specified as symmetric or asymmetric, and in the latter case, the direction of correlation between merging firm prices, if any, can also be specified (see, :class:`mergeron.gen.PriceSpec`).

The market sample may be restricted to mergers meeting the HSR filing requirement under two alternative approaches: in the one, the smaller of the two merging firms meets the HSR filing threshold for the smaller (acquired) firm. In the other, the :math:`n`-th firm's size matches the size requirement for the smaller merging firm (see, :class:`mergeron.gen.SSZConstants`). The second assumption avoids the unfortunate assumption in the first that, within the resulting sample, the larger merging firm be at least 10 times as large as the smaller merging firm, as a consequence of the full definition of the HSR filing requirement.

The full specification of a market sample is given in a :class:`mergeron.gen.market_sample.MarketSample` object, including the above parameters. Data are drawn by invoking :meth:`mergeron.gen.market_sample.MarketSample.generate_sample` which adds a :attr:`data` property of class, :class:`mergeron.gen.MarketDataSample`. Enforcement or clearance counts are computed by invoking :meth:`mergeron.gen.market_sample.MarketSample.estimate_enf_counts`, which adds an :attr:`enf_counts` property of class :class:`mergeron.gen.UPPTestsCounts`. For fast, parallel generation of enforcement or clearance counts over large market data samples that ordinarily would exceed available limits on machine memory, the user can invoke the method :meth:`estimate_enf_counts` on a :class:`mergeron.gen.market_sample.MarketSample` object without first invoking :meth:`generate_sample`. Note, however, that this strategy does not retain the market sample in memory in the interests of conserving memory and maintaining high performance (the user can specify that the market sample and enforcement statistics be stored to permanent storage; when saving to current PCIe NVMe storage, the performance penalty is slight, but can be considerable if saving to SATA storage).

Enforcement statistics based on FTC investigations data and test data are printed to screen or rendered to LaTex files (for processing into publication-quality tables) using methods provided in :mod:`mergeron.gen.enforcement_stats`.

Programs demonstrating the use of this package are included in the sub-package, :mod:`mergeron.demo`.

This package includes  a class, :class:`mergeron.core.pseudorandom_numbers.MulithreadedRNG` for generating random numbers with selected continuous distribution over specified parameters, and with CPU multithreading on machines with multiple virtual, logical, or physical CPU cores. This class is an adaptation from the documentation of the :mod:`numpy` package, from the discussion on `multithreaded random-number generation <https://numpy.org/doc/stable/reference/random/multithreading.html>_`; the version included here permits selection of the distribution with pre-tests to catch and inform on common errors. To access these directly:

.. code-block:: python

    import mergeron.core.pseudorandom_numbers as prng

Documentation for this package is in the form of the API Reference. Documentation for individual functions and classes is accessible within a python shell. For example:

.. code-block:: python

    import mergeron.core.market_sample as market_sample

    help(market_sample.MarketSample)

"Extras" Subpackage
---------------------

The "Extras" sub-package includes a small number of modules potentially useful to users, but which do not implement the principal functions of the package, and are hence considered "extras"  or "external" modules. One of these modules is, in fact, repackaged here although published independently.

On of the external modules provides methods for estimating confidence intervals for proportions and for contrasts (differences) in proportions. This module  improve is coded for conformance to the literature and to results from the corresponding modules in :code:`R`. Although written from scratch, the APIs implemented in the module included here are designed for consistency with the APIs in, :mod:`statsmodels.stats.proportion` from the package, :mod:`statsmodels` (https://pypi.org/project/statsmodels/). To access these directly:

.. code-block:: python

    import mergeron.ext.proportions_tests as prci

Module :mod:`mergeron.ext.xlsxw_helper` is useful for writing highly formatted output to spreadsheets with xlsx format. The class, :class:`mergeron.ext.xlsxw_helper.CFmt` and function, :func:`mergeron.ext.xlsxw_helper.array_to_sheet` are of particular interest, and can be accessed as :code:`xlh.CFmt` and :code:`xlh.array_to_sheet` with the following import:

.. code-block:: python

    import mergeron.ext.xlsxw_helper as xlsxw_helper

A recent version of Paul Tol's python module, :mod:`tol_colors.py`, which provides high-contrast color schemes for making displays with improved visibility for individuals with color-blindness, is redistributed within this package. Other than re-formatting and type annotation, the :mod:`mergeron.ext.tol_colors` module is re-distributed as downloaded from, https://personal.sron.nl/~pault/data/tol_colors.py. The :mod:`tol_colors.py` module is distributed under the Standard 3-clause BSD license. To access the :mod:`mergeron.ext.tol_colors` module directly:

.. code-block:: python

    import mergeron.ext.tol_colors as ptc

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

