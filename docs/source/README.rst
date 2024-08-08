mergeron: Merger Policy Analysis using Python
=============================================

Analyze the sets of mergers conforming to concentration and diversion ratio bounds. Analyze intrinsic enforcement rates, and intrinsic clearance rates, under concentration, diversion ratio, GUPPI, CMCR, and IPR bounds using generated data with specified distributions of market shares, price-cost margins, firm counts, and prices, optionally imposing restrictions implied by statutory filing thresholds and/or Bertrand-Nash oligopoly with MNL demand. Download and analyze merger investigations data published by the U.S. Federal Trade Commission in various reports on extended merger investigations (Second Requests) during 1996 to 2011.

Here, enforcement rates derived with merger enforcement as being exogenous to firm conduct are defined as intrinsic enforcement rates, and similarly intrinsic clearance rates. Depending on the merger enforcement regime, or merger control regime, intrinsic enforcement rates may also not be the complement of intrinsic clearance rates, i.e, it is not necessarily true that the intrinsic clearance rate estimate for a given enforcement regime is 1 minus the intrinsic enforcement rate. In contrast, observed enforcement rates reflect the deterrent effects of merger enforcement on firm conduct as well as the effects of merger screening on the level of enforcement; and, by definition, the observed clearance rate is 1 minus the observed enforcement rate.

Introduction
------------

Module :mod:`.core.guidelines_boundaries` includes classes for specifying concentration bounds (:class:`.core.guidelines_boundaries.ConcentrationBoundary`) and diversion-ratio bounds (:class:`.core.guidelines_boundaries.DiversionRatioBoundary`), with automatic generation of boundary (as an array of share-pairs) and area. This module also includes a function for generating plots of concentration and diversion-ratio boundaries, and functions for mapping GUPPI standards to concentration (ΔHHI) standards, and vice-versa.

Module :mod:`.gen.data_generation` includes the :class:`.gen.data_generation.MarketSample` which provides for a rich specification of shares and diversion ratios (:attr:`.gen.data_generation.MarketSample.share_spec`), margins (:attr:`.gen.data_generation.MarketSample.pcm_spec`, prices (:attr:`.gen.data_generation.MarketSample.price_spec`), and HSR filing requirements (:attr:`.gen.data_generation.MarketSample.hsr_filing_test_type`), and with methods for, (i) generating sample data (:meth:`.gen.data_generation.MarketSample.generate_sample`), and (ii) estimating enforcement or clearance rates under specified enforcement regimes given a method of aggregating diversion ratio or GUPPI estimates for the firms in a merger (:meth:`.gen.data_generation.MarketSample.estimate_enf_counts`). While the latter populate the properties, :attr:`.gen.data_generation.MarketSample.data`
and :attr:`.gen.data_generation.MarketSample.enf_counts`, respectively, the underlying methods for generating standalone :attr:`MarketDataSample` and :attr:`UPPTestCounts` objects are included in the class definition, with helper functions defined in the modules, :mod:`.gen.data_generation_functions` and :mod:`.gen.upp_tests`. Notably, market shares are generated for a sample of markets with firm-count distributed as specified in :attr:`.gen.data_generation.MarketSample.share_spec.firm_count_weights`, with defaults as discussed below (also see, :attr:`.gen.ShareSpec.firm_count_weights`.

By default, merging-firm shares are drawn with uniform distribution over the space :math:`s_1 + s_2 \leqslant 1` for an unspecified number of firms. Alternatively, shares may be drawn from the Dirichlet distribution, with specified shape parameters (see property `dist_type` of :attr:`.gen.data_generation.MarketSample.share_spec`, of type, :class:`.gen.SHRDistribution`). When drawing shares from the Dirichlet distribution, the user specifies the `firm_count_weights` property of :attr:`.gen.data_generation.MarketSample.share_spec`, as a vector of weights specifying the frequency distribution over sequential firm counts, e.g., :attr:`[133, 184, 134, 52, 32, 10, 12, 4, 3]` to specify shares drawn from Dirichlet distributions with 2 to 10 pre-merger firms distributed as in data for FTC merger investigations during 1996--2003 (See, for example, Table 4.1 of `FTC, Horizontal Merger Investigations Data, Fiscal Years 1996--2003 (Revised: August 31, 2004) <https://www.ftc.gov/sites/default/files/documents/reports/horizontal-merger-investigation-data-fiscal-years-1996-2003/040831horizmergersdata96-03.pdf>`_). If the property `firm_count_weights` is not explicitly assigned a value when defining :attr:`.gen.data_generation.MarketSample.share_spec`, the default values is used, which results in a sample of markets with 2 to 7 firms with relative frequency in inverse proportion to firm-count, with 2-firm markets being 6 times as likely to be drawn as 7-firm markets.

Recapture rates can be specified as, "proportional", "inside-out", "outside-in" (see :class:`.RECForm`). The "inside-out" specification (assigning :attr:`.RECForm.INOUT` to the `recapture_form` property of :attr:`.gen.data_generation.MarketSample.share_spec`) results in recapture ratios consistent with merging-firms' in-market shares and a default recapture rate. The "outside-in" specification (assigning :attr:`.RECForm.INOUT` to the `recapture_form` property of :attr:`.gen.data_generation.MarketSample.share_spec`) yields diversion ratios from purchase probabilities drawn at random for :math:`N+1` goods, from which are derived market shares and recapture rates for the :math:`N` goods in the putative market (see, :class:`.gen.ShareSpec`). The "outside-in" specification is invalid when the distribution of markets over firm-count is unspecified, i.e., when the property `dist_type` of :attr:`.gen.data_generation.MarketSample.share_spec` is assigned :attr:`.gen.ShareDistributions.UNI`, raising a :mod:`ValueError` exception. The "proportional" form (`recapture_form` = :attr:`.RECForm.FIXED`) is often used in the literature, as an approximation to the "inside-out" calibration. See, for example, :ref:`Coate (2011) <coate2011>`.

Price-cost-margins may be specified as having uniform distribution, Beta distribution (including a bounded Beta distribution with specified mean and variance), or an empirical distribution (see, :class:`.gen.PCMSpec`). The empirical margin distribution is based on resampling margin data published by Prof. Damodaran of NYU Stern School of Business (see Notes), using an estimated Gaussian KDE. The second merging firm's margin (per the property `firm2_pcm_constraint` of :attr:`.gen.data_generation.MarketSample.pcm_spec`) may be specified as symmetric, i.i.d., or subject to equilibrium conditions for (profit-maximization in) Bertrand-Nash oligopoly with MNL demand (:class:`.gen.FM2Constraint`).

Prices may be specified as symmetric or asymmetric, and in the latter case, the direction of correlation between merging firm prices, if any, can also be specified (see, :class:`.gen.PriceSpec`). Prices may also be defined by imposing cost symmetry on firms in the sample, with fixed unit marginal costs normalized to 1 unit, such that price equal :math:`1 / (1 - \pmb{m})`, where :math:`\pmb{m}` represents the array of margins for firms in the sample.

The market sample may be restricted to mergers meeting the HSR filing requirement under two alternative approaches: in the one, the smaller of the two merging firms meets the lower HSR size threshold ($10 million, as adjusted) and the larger of the two merging firms meets the size test if it's share is no less than 10 times the share of the smaller firm. In the other, the :math:`n`-th firm's size is maintained as $10 million, as adjusted (see, :class:`.gen.SSZConstant`), and a merger meets the HSR filing test if either, (a.) the smaller merging firm is no smaller than the n-th firm and the larger merging firm is at 10-times as large as the n-th firm, or (b.) the smaller merging firm's market share is in excess of 10%; in effect this version of the test maintains that if the smaller merging firm's market share exceeds 10%, the value of the transaction exceeds $200 million, as adjusted, and the size-of-person test is eliminated (see, :ref:`FTC (2008, p. 12) <ftc_premerger_guide2>`; the above are simplifications of the statutory HSR filing requirements). The second assumption avoids the unfortunate assumption in the first that, within the resulting sample, the larger merging firm be at least 10 times as large as the smaller merging firm, as a consequence of the full definition of the HSR filing requirement.

The full specification of a market sample is given in a :class:`.gen.data_generation.MarketSample` object, including the above parameters. Data are drawn by invoking :meth:`.gen.data_generation.MarketSample.generate_sample` which adds a :attr:`data` property of class, :class:`.gen.MarketDataSample`. Enforcement or clearance counts are computed by invoking :meth:`.gen.data_generation.MarketSample.estimate_enf_counts`, which adds an :attr:`enf_counts` property of class :class:`.gen.UPPTestsCounts`. For fast, parallel generation of enforcement or clearance counts over large market data samples that ordinarily would exceed available limits on machine memory, the user can invoke the method :meth:`.gen.data_generation.MarketSample.estimate_enf_counts` on a :class:`.gen.data_generation.MarketSample` object without first invoking :meth:`.gen.data_generation.MarketSample.generate_sample`. Note, however, that this strategy does not retain the market sample in memory in the interests of conserving memory and maintaining high performance (the user can specify that the market sample and enforcement statistics be stored to permanent storage; when saving to current PCIe NVMe storage, the performance penalty is slight, but can be considerable if saving to SATA storage).

Enforcement statistics based on FTC investigations data and test data are printed to screen or rendered to LaTex files (for processing into publication-quality tables) using methods provided in :mod:`.gen.enforcement_stats`.

Programs demonstrating the use of this package are included in the sub-package, :mod:`.demo`.

This package includes  a class, :class:`.core.pseudorandom_numbers.MulithreadedRNG` for generating random numbers with selected continuous distribution over specified parameters, and with CPU multithreading on machines with multiple virtual, logical, or physical CPU cores. This class is an adaptation from the documentation of the :mod:`numpy` package, from the discussion on `multithreaded random-number generation <https://numpy.org/doc/stable/reference/random/multithreading.html>_`; the version included here permits selection of the distribution with pre-tests to catch and inform on common errors. To access these directly:

.. code-block:: python

    import mergeron.core.pseudorandom_numbers as prng

Documentation for this package is in the form of the API Reference. Documentation for individual functions and classes is accessible within a python shell. For example:

.. code-block:: python

    import mergeron.core.market_sample as market_sample

    help(market_sample.MarketSample)

.. rubric:: References

.. _coate2011:

Coate, M. B. (2011). Benchmarking the upward pricing pressure model with Federal Trade
Commission evidence. Journal of Competition Law & Economics, 7(4), 825--846. URL: https://doi.org/10.1093/joclec/nhr014.

.. _ftc_premerger_guide2:

FTC Premerger Notification Office. “To File or Not to File: When You Must File a Premerger Notification Report Form”. 2008 (September, revised). URL: https://www.ftc.gov/sites/default/files/attachments/premerger-introductory-guides/guide2.pdf


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

