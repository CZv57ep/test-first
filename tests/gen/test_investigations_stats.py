import mergeron.core.ftc_merger_investigations_data as fid
import mergeron.gen.investigations_stats as clstl
import numpy as np
import pytest
from mergeron.core.ftc_merger_investigations_data import TableData  # noqa F401
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

invdata_array_dict = fid.construct_invdata(
    fid.invdata_dump_path,
    flag_backward_compatibility=False,
    flag_pharma_for_exclusion=True,
)


@pytest.mark.parametrize(
    "_stats_group, _test_val",
    zip(
        (clstl.StatsGrpSelector.FC, clstl.StatsGrpSelector.DL),
        np.array([[573, 132], [780, 173]]),
        strict=True,
    ),
)
def test_clrenf_stats(
    _stats_group: clstl.StatsGrpSelector, _test_val: NDArray[np.int_]
) -> None:
    _clrenf_sel = clstl.CLRENFSelector.CLRN
    _clrenf_stats_cnts = clstl.clrenf_stats_cnts_by_group(
        invdata_array_dict,
        "1996-2003",
        clstl.INDGRPConstants.ALL,
        clstl.EVIDENConstants.UR,
        _stats_group,
        _clrenf_sel,
    )[:, -2:]
    _clrenf_stats_totals = np.einsum("ij->j", _clrenf_stats_cnts)
    assert_array_equal(_clrenf_stats_totals, _test_val)


clrenf_sel = clstl.CLRENFSelector.CLRN
# Test print functionality:
for data_period in "1996-2003", "2004-2011":
    for evid_class in clstl.EVIDENConstants.UR, clstl.EVIDENConstants.ED:
        for stats_group in clstl.StatsGrpSelector:
            if stats_group == clstl.StatsGrpSelector.HD:
                continue

            for return_type in clstl.StatsReturnSelector:
                (clrenf_stats_hdr_list, clrenf_stats_dat_list) = (
                    clstl.clrenf_stats_output(
                        invdata_array_dict,
                        data_period,
                        clstl.INDGRPConstants.ALL,
                        evid_class,
                        stats_group,
                        clrenf_sel,
                        return_type_sel=return_type,
                        sort_order=(
                            clstl.SortSelector.UCH
                            if stats_group == clstl.StatsGrpSelector.FC
                            else clstl.SortSelector.REV
                        ),
                    )
                )
