import gc

import mergeron.core.pseudorandom_numbers as rmp
import numpy as np
import pytest
from mergeron import RECConstants
from mergeron.gen import (
    FCOUNT_WTS_DEFAULT,
    FM2Constants,
    PCMConstants,
    PCMSpec,
    ShareSpec,
    SHRConstants,
)
from mergeron.gen.market_sample import MarketSample
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

tvals_dict = {
    # Test with uniform distribution (unrestricted shares), proportional recapture spec
    (SHRConstants.UNI, RECConstants.FIXED, FM2Constants.IID): np.array([
        0.3333775494826034,
        0.4000546725161907,
        0.16666859164821712,
    ]),
    # Test with uniform distribution (unrestricted shares), inside-out recapture spec,
    # .i.i.d PCM values
    (SHRConstants.UNI, RECConstants.INOUT, FM2Constants.IID): np.array([
        0.3333775494826034,
        0.37381043401375175,
        0.16666859164821712,
    ]),
    # Test with uniform distribution (unrestricted shares), inside-out recapture spec,
    # MNL-consistent PCM values
    (SHRConstants.UNI, RECConstants.INOUT, FM2Constants.MNL): np.array([
        0.32502286465475816,
        0.36698399533280657,
        0.16728195834397885,
    ]),
    # Test with flat dirichlet, proportional recapture spec, i.i.d. PCM values
    (SHRConstants.DIR_FLAT, RECConstants.FIXED, FM2Constants.IID): np.array([
        0.34331471002492747,
        0.4639299532812615,
        0.18757732533817648,
        0.6866050342033266,
        0.34341223113209834,
        3.3331078,
        6.0,
    ]),
    # Test with flat dirichlet, inside-out recapture spec, i.i.d. PCM values
    (SHRConstants.DIR_FLAT, RECConstants.INOUT, FM2Constants.IID): np.array([
        0.34331471002492747,
        0.40745812736302944,
        0.18757732533817648,
        0.6866050342033266,
        0.34341223113209834,
        3.3331078,
        6.0,
    ]),
    # Test with flat dirichlet, inside-out recapture spec, MNL-consistent PCM values
    (SHRConstants.DIR_FLAT, RECConstants.INOUT, FM2Constants.MNL): np.array([
        0.3292729603184209,
        0.3932420089295269,
        0.1849591426831523,
        0.6644936417432625,
        0.3100273072246238,
        3.4231814,
        6.0,
    ]),
    # Test with flat dirichlet, outside-in recapture spec, i.i.d PCM values
    (SHRConstants.DIR_FLAT, RECConstants.OUTIN, FM2Constants.IID): np.array([
        0.34332714606238723,
        0.34327732941767747,
        0.1876114023527494,
        0.6866412416605577,
        0.3434294542126609,
        3.3331078,
        6.0,
    ]),
    # Test with unweighted flat dirichlet, proportional recapture spec, i.i.d PCM values
    (SHRConstants.DIR_FLAT_CONSTR, RECConstants.FIXED, FM2Constants.IID): np.array([
        0.34331471002492747,
        0.4639299532812615,
        0.18757732533817648,
        0.6866050342033266,
        0.34341223113209834,
        3.3331078,
        6.0,
    ]),
    (SHRConstants.DIR_ASYM, RECConstants.FIXED, FM2Constants.IID): np.array([
        0.3433240778115128,
        0.46397739836107904,
        0.21945448919130042,
        0.6521636462229494,
        0.3432975275484357,
        3.3331078,
        6.0,
    ]),
    (SHRConstants.DIR_COND, RECConstants.FIXED, FM2Constants.IID): np.array([
        0.4444223338483846,
        0.6475767300577249,
        0.3372855445979423,
        0.8355498773732144,
        0.2380192634146831,
        3.3331078,
        6.0,
    ]),
}


@pytest.mark.parametrize("_test_parms, _test_array", tuple(tvals_dict.items()))
def test_gen_market_sample(
    _test_parms: tuple[SHRConstants, RECConstants, FM2Constants],
    _test_array: NDArray[np.float64],
    _tcount: int = 10**7,
    _nthreads: int = 16,
) -> None:
    (_mktshr_dist_type_test, _recapture_form_test, _pcm_dist_firm2_test) = _test_parms
    # Reinitialize the seed sequence for each test run
    #   (this makes the tests invariant to the order in which run)
    _rng_seed_seq_tup = rmp.gen_seed_seq_list_default(
        2 if _mktshr_dist_type_test == SHRConstants.UNI else 3
    )

    _mkt_sample = MarketSample(
        pcm_spec=PCMSpec(_pcm_dist_firm2_test, PCMConstants.UNI, None)
    )
    if _mktshr_dist_type_test == SHRConstants.UNI:
        _shr_dist_parms = None
        _fcount_weights = None
        _test_func = _tfunc_sample_with_unrestricted_shares
    else:
        # np.arange(1, 6)[::-1] / np.arange(1, 6).sum()
        _fcount_weights = FCOUNT_WTS_DEFAULT
        _shr_dist_parms = None
        _test_func = _tfunc_sample_with_dirichlet_shares

    _mkt_sample.share_spec = ShareSpec(
        _recapture_form_test,
        None if _recapture_form_test == RECConstants.OUTIN else 0.80,
        _mktshr_dist_type_test,
        _shr_dist_parms,
        _fcount_weights,
    )
    _array_to_test = _test_func(_tcount, _mkt_sample, _rng_seed_seq_tup, _nthreads)

    print(
        f"{_mktshr_dist_type_test}, {_recapture_form_test} ({_pcm_dist_firm2_test}): {_tcount:,d}",
        repr(_array_to_test),
        sep="\n",
    )

    # assert_array_equal((0, 0), (0, 0))
    # if _pcm_dist_firm2_test != FM2Constants.MNL:
    assert_array_equal(_array_to_test, _test_array)
    del _mkt_sample
    gc.collect()


def _tfunc_sample_with_unrestricted_shares(
    _sample_size: int,
    _mkt_sample: MarketSample,
    _rng_seed_seq_tup: list[np.random.SeedSequence],
    _nthreads: int,
    /,
) -> NDArray[np.float64 | np.float64]:
    _mkt_sample.generate_sample(
        sample_size=_sample_size, seed_seq_list=_rng_seed_seq_tup, nthreads=_nthreads
    )
    return np.array([
        _mkt_sample.data.frmshr_array.mean(),
        _mkt_sample.data.divr_array.mean(),
        _mkt_sample.data.hhi_delta.mean(),
    ])


def _tfunc_sample_with_dirichlet_shares(
    _sample_size: int,
    _mkt_sample: MarketSample,
    _rng_seed_seq_tup: list[np.random.SeedSequence],
    _nthreads: int,
    /,
) -> NDArray[np.float64 | np.float64]:
    _mkt_sample.generate_sample(
        sample_size=_sample_size, seed_seq_list=_rng_seed_seq_tup, nthreads=_nthreads
    )
    return np.array((
        _mkt_sample.data.frmshr_array.mean(),
        _mkt_sample.data.divr_array.mean(),
        _mkt_sample.data.hhi_delta.mean(),
        _mkt_sample.data.hhi_post.mean(),
        _mkt_sample.data.nth_firm_share.mean(),
        _mkt_sample.data.fcounts.mean(),
        _mkt_sample.data.fcounts.max(),
    ))


if __name__ == "__main__":
    np.set_printoptions(precision=18)

    print(
        "This module defines functions useful for generating data",
        "for modeling antitrust and merger analysis.",
    )
    print()

    for _test_parms, _test_array in tvals_dict.items():
        test_gen_market_sample(_test_parms, _test_array)
