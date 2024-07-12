import numpy as np
from icecream import ic  # type: ignore
from mergeron.core.pseudorandom_numbers import (
    MultithreadedRNG,
    gen_seed_seq_list_default,
)
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)


def test_mrng_dirichlet(_tcount: int = 10**8, _fcount: int = 5) -> None:
    """Test multithreaded generation of Dirichlet variates"""

    ic("Test multithreaded generation of Dirichlet variates")
    _test_out = np.empty((_tcount, _fcount), dtype=np.float64)
    _mrng = MultithreadedRNG(
        _test_out,
        dist_type="Dirichlet",
        dist_parms=np.ones(_fcount),
        seed_sequence=gen_seed_seq_list_default(1)[0],
        nthreads=16,
    )
    _mrng.fill()
    ic(_test_out)
    ic(_test_out.mean(axis=0))
    assert_array_equal(
        _test_out.mean(axis=0),
        np.array([
            0.1999916675222448,
            0.20000937237277838,
            0.20000280540828835,
            0.2000040263284762,
            0.1999921283682021,
        ]),
    )
    assert_array_almost_equal(
        _test_out.mean(axis=0),
        np.array([0.200] * _fcount),
        decimal=int(np.log10(_tcount) / 2),
    )
    assert_array_equal(_test_out.shape, (_tcount, _fcount))
    assert_equal(np.round(_test_out.sum()), _tcount)
    del _test_out, _mrng


def test_mrng_beta(_tcount: int = 10**8, _fcount: int = 5) -> None:
    """Test multithreaded generation of Beta variates"""

    ic("Test multithreaded generation of Beta variates")
    _test_out = np.empty((_tcount, _fcount), dtype=np.float64)
    _mrng = MultithreadedRNG(
        _test_out,
        dist_type="Beta",
        dist_parms=np.ones(2),
        seed_sequence=gen_seed_seq_list_default(1)[0],
        nthreads=16,
    )
    _mrng.fill()
    ic(_test_out.mean(axis=0))
    assert_array_equal(
        _test_out.mean(axis=0),
        np.array([
            0.4999797742786088,
            0.5000255089039324,
            0.500004827320672,
            0.5000165032197761,
            0.49997575795924837,
        ]),
    )
    assert_array_almost_equal(
        _test_out.mean(axis=0),
        np.array([0.500] * _fcount),
        decimal=int(np.log10(_tcount) / 2),
    )
    assert_array_equal(_test_out.shape, (_tcount, _fcount))
    del _test_out, _mrng

    ic("Test multithreaded generation of Scaled Beta variates")
    _test_out = np.empty((_tcount, 1), dtype=np.float64)
    _beta_mu, _beta_sigma = [0.5, 0.28867513459481287]
    _mul = np.divide(_beta_mu - _beta_mu**2 - _beta_sigma**2, _beta_sigma**2)
    _dist_parms_beta = np.array(
        [_beta_mu * _mul, (1 - _beta_mu) * _mul], dtype=np.float64
    )
    _mrng = MultithreadedRNG(
        _test_out,
        dist_type="Beta",
        dist_parms=_dist_parms_beta,
        seed_sequence=gen_seed_seq_list_default(1)[0],
        nthreads=16,
    )
    _mrng.fill()
    ic(_test_out.mean())
    assert_equal(_test_out.mean(), 0.5000134457423757)
    assert_almost_equal(_test_out.mean(), 0.500, decimal=int(np.log10(_tcount) / 2))
    del _test_out, _mrng


if __name__ == "__main__":
    test_mrng_dirichlet()
    test_mrng_beta()
