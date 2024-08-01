import mergeron.core.ftc_merger_investigations_data as fid
import mergeron.gen.enforcement_stats as esl
import numpy as np
import pytest
from mergeron.gen import INVResolution
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
        (esl.StatsGrpSelector.FC, esl.StatsGrpSelector.DL),
        np.array([[573, 132], [780, 173]]),
        strict=True,
    ),
)
def test_enf_stats(
    _stats_group: esl.StatsGrpSelector, _test_val: ArrayBIGINT
) -> None:
    _enf_spec = INVResolution.CLRN
    _enf_stats_cnts = esl.enf_stats_listing_by_group(
        invdata_array_dict,
        "1996-2003",
        esl.INDGRPConstants.ALL,
        esl.EVIDENConstants.UR,
        _stats_group,
        _enf_spec,
    )[:, -2:]
    _enf_stats_totals = np.einsum("ij->j", _enf_stats_cnts)
    assert_array_equal(_enf_stats_totals, _test_val)


# enf_spec = INVResolution.CLRN
# # Test print functionality:
# for data_period in "1996-2003", "2004-2011":
#     for evid_class in esl.EVIDENConstants.UR, esl.EVIDENConstants.ED:
#         for stats_group in esl.StatsGrpSelector:
#             if stats_group == esl.StatsGrpSelector.HD:
#                 continue

#             for return_type in esl.StatsReturnSelector:
#                 (enf_stats_hdr_list, enf_stats_dat_list) = (
#                     esl.enf_stats_output(
#                         invdata_array_dict,
#                         data_period,
#                         esl.INDGRPConstants.ALL,
#                         evid_class,
#                         stats_group,
#                         enf_spec,
#                         return_type_sel=return_type,
#                         sort_order=(
#                             esl.SortSelector.UCH
#                             if stats_group == esl.StatsGrpSelector.FC
#                             else esl.SortSelector.REV
#                         ),
#                     )
#                 )
