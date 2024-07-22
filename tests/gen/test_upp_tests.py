import gc
from datetime import datetime, timedelta

import mergeron.core.guidelines_boundaries as gbl
import mergeron.core.pseudorandom_numbers as rmp
import mergeron.gen.enforcement_stats as esl
import numpy as np
import re2 as re  # type: ignore
from mergeron import RECConstants, UPPAggrSelector
from mergeron.gen import INVResolution, ShareSpec, SHRConstants, UPPTestRegime
from mergeron.gen.market_sample import MarketSample

teststr_pat = re.compile(r"(?m)^ +")
stats_sim_byfirmcount_teststr = teststr_pat.sub(
    "",
    R"""
    {2 to 1} & 23,581,830 & 132,308 &     0 & 1,321 \\
    {3 to 2} & 32,630,452 & 2,914,461 & 1,493,365 & 459,249 \\
    {4 to 3} & 23,756,506 & 4,131,208 & 2,706,564 & 736,150 \\
    {5 to 4} & 9,216,234 & 2,344,766 & 1,758,769 & 450,268 \\
    {6 to 5} & 5,672,777 & 1,864,793 & 1,523,661 & 378,081 \\
    {7 to 6} & 1,773,016 & 703,304 & 608,731 & 149,817 \\
    {8 to 7} & 2,128,579 & 975,146 & 877,432 & 216,329 \\
    {9 to 8} & 708,913 & 363,684 & 336,695 & 84,069 \\
    {10 to 9} & 531,693 & 299,188 & 282,775 & 71,885 \\
    TOTAL & 100,000,000 & 13,728,858 & 9,587,992 & 2,547,169 \\
    """,
).lstrip()

stats_sim_bydelta_teststr = teststr_pat.sub(
    "",
    R"""
    {[2500, 5000]} & 27,899,813 & 210,013 &     0 & 5,948 \\
    {[1200, 2500)} & 20,815,469 & 472,156 &     0 & 39,671 \\
    {[800, 1200)} & 9,905,844 & 496,873 &     0 & 53,114 \\
    {[500, 800)} & 9,806,064 & 880,669 & 168,069 & 102,438 \\
    {[300, 500)} & 8,587,770 & 1,410,491 & 631,791 & 163,647 \\
    {[200, 300)} & 5,395,963 & 1,374,310 & 890,373 & 168,277 \\
    {[100, 200)} & 6,759,638 & 2,469,256 & 1,978,386 & 341,934 \\
    {[0, 100)} & 10,829,439 & 6,415,090 & 5,919,373 & 1,672,140 \\
    TOTAL & 100,000,000 & 13,728,858 & 9,587,992 & 2,547,169 \\
    """,
).lstrip()

stats_sim_byconczone_teststr = teststr_pat.sub(
    "",
    R"""
    \node[align = left, fill=VibrRed] {Red Zone (SLC Presumption)}; & 81,399,163 & 4,586,532 & 1,484,261 & 501,049 \\
    \node[align = left, fill=HiCoYellow] {Yellow Zone}; & 7,554,220 & 2,629,117 & 2,088,034 & 361,314 \\
    \node[text = HiCoYellow, fill = white, align = right] { \(HHI_{post} \geqslant \text{2400 pts. and } \Delta HHI \in \text{[100, 200) pts.}\) }; & 6,056,943 & 2,130,580 & 1,663,925 & 296,260 \\
    \node[text = HiCoYellow, fill = white, align = right] { \(HHI_{post} \in \text{[1800, 2400) pts. and } \Delta HHI \geqslant \text{100 pts.}\) }; & 1,497,277 & 498,537 & 424,109 & 65,054 \\
    \node[align = left, fill=BrightGreen] {Green Zone (Safeharbor)}; & 11,046,617 & 6,513,209 & 6,015,697 & 1,684,806 \\
    \node[text = BrightGreen, fill = white, align = right] { \(HHI_{post} \geqslant \text{2400 pts. and } \Delta HHI < \text{100 pts.}\) }; & 9,444,666 & 5,437,272 & 4,969,286 & 1,422,065 \\
    \node[text = BrightGreen, fill = white, align = right] { \(HHI_{post} \in \text{[1800, 2400) pts. and } \Delta HHI < \text{100 pts.}\) }; & 1,075,970 & 748,603 & 722,581 & 193,422 \\
    \node[text = BrightGreen, fill = white, align = right] { \(HHI_{post} < \text{1800 pts.}\) }; & 525,981 & 327,334 & 323,830 & 69,319 \\
    \node[align = left, fill=OBSHDRFill] {TOTAL}; & 100,000,000 & 13,728,858 & 9,587,992 & 2,547,169 \\
    """,
).lstrip()

stats_sim_bydelta_unrestricted_teststr = teststr_pat.sub(
    "",  # Change 30.2 to 30.3 in 2d row
    R"""
    {[0, 100)} &    6.3\% &   46.6\% &   39.4\% \\
    {[100, 200)} &    4.9\% &   30.3\% &   21.0\% \\
    {[200, 300)} &    4.4\% &   22.3\% &   12.0\% \\
    {[300, 500)} &    7.8\% &   15.1\% &    5.4\% \\
    {[500, 800)} &   10.1\% &    8.5\% &    1.3\% \\
    {[800, 1200)} &   11.6\% &    5.0\% &    0.0\% \\
    {[1200, 2500)} &   28.4\% &    2.4\% &    0.0\% \\
    {[2500, 5000]} &   26.6\% &    1.0\% &    0.0\% \\
    TOTAL  &  100.0\% &    8.9\% &    4.6\% \\
    """,
).lstrip()


def test_clearance_rate_calcs() -> None:
    _test_sel: UPPTestRegime = UPPTestRegime(
        INVResolution.CLRN, UPPAggrSelector.MAX, None
    )

    _mkt_sample = MarketSample(
        share_spec=ShareSpec(
            RECConstants.FIXED,
            0.80,
            SHRConstants.DIR_FLAT,
            None,  # TODO: type-fix this, with None as default
            np.array((133, 184, 134, 52, 32, 10, 12, 4, 3)),
        )
    )

    _start_time = datetime.now()
    _mkt_sample.estimate_enf_counts(
        gbl.GuidelinesThresholds(2010).safeharbor,
        _test_sel,
        sample_size=10**8,
        seed_seq_list=rmp.gen_seed_seq_list_default(3),
        nthreads=16,
    )
    # upp_tests_counts = utl.sim_enf_cnts_ll(
    #     _mkt_sample_spec,
    #     gbl.GuidelinesThresholds(2010).safeharbor,
    #     seed_seq_list=rmp.gen_seed_seq_list_default(3),
    #     sim_test_regime=_test_sel,
    #     nthreads=16,
    # )
    _total_duration = datetime.now() - _start_time

    print(
        f"Estimations completed in total duration of {_total_duration / timedelta(seconds=1):.6f} secs."
    )

    upp_tests_counts = _mkt_sample.enf_counts
    _return_type_sel = esl.StatsReturnSelector.CNT
    print()
    print(
        f"Simulated {_test_sel.resolution.capitalize()} stats by number of significant competitors:"
    )
    _stats_hdr_list, _stats_dat_list = esl.latex_tbl_enf_stats_1dim(
        upp_tests_counts.by_firm_count[:, :-1], return_type_sel=_return_type_sel
    )

    _stats_byfirmcount_teststr_val = "".join([
        "{} & {} {}".format(
            _stats_hdr_list[g], " & ".join(_stats_dat_list[g]), esl.LTX_ARRAY_LINEEND
        )
        for g in range(len(_stats_hdr_list))
    ])
    print(_stats_byfirmcount_teststr_val)
    del _stats_hdr_list, _stats_dat_list

    print()
    print(f"Simulated {_test_sel.resolution.capitalize()} stats by range of ∆HHI")
    _stats_hdr_list, _stats_dat_list = esl.latex_tbl_enf_stats_1dim(
        upp_tests_counts.by_delta[:, :-1],
        return_type_sel=_return_type_sel,
        sort_order=esl.SortSelector.REV,
    )

    _stats_bydelta_teststr_val = "".join([
        "{} & {} {}".format(
            _stats_hdr_list[g], " & ".join(_stats_dat_list[g]), esl.LTX_ARRAY_LINEEND
        )
        for g in range(len(_stats_hdr_list))
    ])
    print(_stats_bydelta_teststr_val)
    del _stats_hdr_list, _stats_dat_list

    print()
    print(
        f"Merger {_test_sel.resolution.capitalize()} stats by Merger Guidelines concentration presumptions"
    )
    _stats_hdr_list, _stats_dat_list = esl.latex_tbl_enf_stats_byzone(
        upp_tests_counts.by_conczone[:, :-1],
        return_type_sel=_return_type_sel,
        sort_order=esl.SortSelector.REV,
    )
    _stats_byzone_teststr_val = "".join([
        "{} & {} {}".format(
            _stats_hdr_list[g], " & ".join(_stats_dat_list[g]), esl.LTX_ARRAY_LINEEND
        )
        for g in range(len(_stats_hdr_list))
    ])
    print(_stats_byzone_teststr_val)
    del _stats_hdr_list, _stats_dat_list

    # Repeatability test:
    if all((
        stats_sim_byfirmcount_teststr == _stats_byfirmcount_teststr_val,
        stats_sim_bydelta_teststr == _stats_bydelta_teststr_val,
        stats_sim_byconczone_teststr == _stats_byzone_teststr_val,
    )):
        print("Tests passed for generation of firm-count-weighted market data.")
    else:
        raise AssertionError(
            "Tests FAILED for generation of firm-count-weighted market data."
        )

    gc.collect()


if __name__ == "__main__":
    test_clearance_rate_calcs()
