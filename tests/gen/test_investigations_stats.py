import mergeron.core.ftc_merger_investigations_data as fid
import mergeron.gen.investigations_stats as isl
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

invdata_array_dict = fid.construct_data(
    fid.INVDATA_ARCHIVE_PATH,
    flag_backward_compatibility=False,
    flag_pharma_for_exclusion=True,
)


@pytest.mark.parametrize(
    "_stats_group, _test_val",
    zip(
        (isl.StatsGrpSelector.FC, isl.StatsGrpSelector.DL),
        np.array([[573, 132], [780, 173]]),
        strict=True,
    ),
)
def test_inv_stats(
    _stats_group: isl.StatsGrpSelector, _test_val: NDArray[np.int_]
) -> None:
    _inv_sel = isl.PolicySelector.CLRN
    _inv_stats_cnts = isl.inv_stats_cnts_by_group(
        invdata_array_dict,
        "1996-2003",
        isl.INDGRPConstants.ALL,
        isl.EVIDENConstants.UR,
        _stats_group,
        _inv_sel,
    )[:, -2:]
    _inv_stats_totals = np.einsum("ij->j", _inv_stats_cnts)
    assert_array_equal(_inv_stats_totals, _test_val)


inv_sel = isl.PolicySelector.CLRN
# Test print functionality:
for data_period in "1996-2003", "2004-2011":
    for evid_class in isl.EVIDENConstants.UR, isl.EVIDENConstants.ED:
        for stats_group in isl.StatsGrpSelector:
            if stats_group == isl.StatsGrpSelector.HD:
                continue

            for return_type in isl.StatsReturnSelector:
                (inv_stats_hdr_list, inv_stats_dat_list) = isl.inv_stats_output(
                    invdata_array_dict,
                    data_period,
                    isl.INDGRPConstants.ALL,
                    evid_class,
                    stats_group,
                    inv_sel,
                    return_type_sel=return_type,
                    sort_order=(
                        isl.SortSelector.UCH
                        if stats_group == isl.StatsGrpSelector.FC
                        else isl.SortSelector.REV
                    ),
                )
