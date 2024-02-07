mergeron: Merger Policy Analysis with Python
============================================

Model the enforcement standards of U.S. Horizontal Merger Guidelines in terms of the sets of mergers conforming to a given Guidelines standard. Analyze intrinsic clearance rates and intrinsic enforcement rates under Guidelines standards, i.e., rates derived from applying a theoretically-derived model to generated data for specified distributions of relevant economic factors. Download and analyze merger investigations data published by FTC, covering investigations during the years, 1996 to 2011.

Intrinsic clearance and enforcement rates are distinct from observed clearance and enforcement rates in that the former do not reflect the effects of screening and deterrence.

Modules of primary interest
---------------------------

:code:`mergeron.core.guidelines_standards` 
    Routines for plotting boundaries of (sets of mergers falling within) specified concentration and diversion ratio boundaries and for calibrating GUPPI thresholds to concentration (ΔHHI) thresholds, and vice-versa. 

:code:`mergeron.gen.data_generation`
    Routines for generating industry data under various distributions of shares, prices, and margins, with and without equilibrium restrictions from MNL demand. The user can specify whether recapture ratios are specified as, "proportional", "inside-out" (i.e., consistent with merging-firms' in-market shares and a default), or "outside-in", (i.e., purchase probabilities are drawn at random for `n+1` goods, from which are derived market shares for the `n` goods in the putative market). Price-cost-margins may be specified as symmetric, i.i.d, or consistent with Bertrand-Nash oligopoly with MNL demand. Prices may be specified as symmetric or asymmetric, and in the latter case, the direction of correlation between merging firm prices, if any, can also be specified.
        
:code:`mergeron.gen.guidelines_tests`
    Routines for testing generated industry data against criteria on gross upward pricing pressure ("GUPPI"), diversion ratio, critical marginal cost reduction ("CMCR"), and indicative price rise ("IPR")/partial merger simulation. Test data are constructed in parallel and the user can specify number of `threads' and sub-sample size for each thread to manage CPU and memory utilization.
        
:code:`mergeron.core.ftc_merger_investigations_data` 
    Routines for downloading FTC merger investigtions data published in 2004, 2007, 2008, and 2013. This module includes a routine for constructing investigations data for non-overlapping periods, and other partitions on the data, subject to the constraints of the reported data.
    
:code:`mergeron.examples`
    Programs for replicating results in a selection papers written using this package.

This package also exposes routines employed for generating random numbers with selected continuous distribution over specified parameters, and with CPU multithreading on machines with multiple virtual, logical, or physical CPU cores. To access these directly:

    import mergeron.core.pseudorandom_numbers as prng

Also included are routines for estimating confidence intervals for proportions and for contrasts (differences) in proportions. To access these directly:

    import mergeron.core.proportions_tests as prci

A recent version of Paul Tol's python module, :code:`tol_colors.py` is redistributed within this package. Other than re-formatting and type annotation, the :code:`tol_colors` module is re-distributed as downloaded from, https://personal.sron.nl/~pault/data/tol_colors.py. The tol_colors.py module is distributed under the Standard 3-clause BSD license. To access the tol_colors module directly:

    import mergeron.ext.tol_colors


.. toctree::
   :caption: Contents
   :maxdepth: 4
   
   modules
   

.. toctree::
    :caption: License
    :maxdepth: 1

    license
    
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
