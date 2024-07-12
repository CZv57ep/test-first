"""
Functions for generating synthetic data under specified distributions.

Uses multiple CPUs when available, with PCG64DXSM as the PRNG
https://github.com/numpy/numpy/issues/16313.

"""

import concurrent.futures
from collections.abc import Sequence
from multiprocessing import cpu_count
from typing import Literal

import numpy as np
from numpy.random import PCG64DXSM, Generator, SeedSequence
from numpy.typing import NDArray

from .. import VERSION  # noqa: TID252

__version__ = VERSION

NTHREADS = 2 * cpu_count()
DIST_PARMS_DEFAULT = np.array([0.0, 1.0], np.float64)


def prng(_s: SeedSequence | None = None, /) -> np.random.Generator:
    """Adopt the PCG64DXSM bit-generator, the future default in numpy.default_rng().

    Parameters
    ----------
    _s
        SeedSequence, for generating random numbers in repeatable fashion.

    Returns
    -------
        A numpy random BitGenerator.

    """
    return Generator(PCG64DXSM(_s))


def gen_seed_seq_list_default(
    _sseq_list_len: int = 3, /, *, generated_entropy: Sequence[int] | None = None
) -> list[SeedSequence]:
    """
    Return specified number of SeedSequences, for generating random variates

    Initializes a specified number of SeedSequences based on a set of
    10 generated "seeds" in a hard-coded list. If the required number of
    random variates is larger than 10, the user must first generate
    a sufficient number of seeds to draw upon for initializing SeedSequences.
    The generated seeds can be reused in subsequent calls to this function.

    Parameters
    ----------
    _sseq_list_len
        Number of SeedSequences to initialize

    generated_entropy
        A list of integers with length not less than _s, to be used as seeds
        for initializing SeedSequences. A list of 10 appropriately generated
        integers is used as default.

    Returns
    -------
        A list of numpy SeedSequence objects, which can be used to seed prng() or to spawn
        seed sequences that can be used as seeds to generate non-overlapping streams in parallel.

    Raises
    ------
    ValueError
        When, :math:`\\_sseq\\_list\\_len > max(10, len(generated\\_entropy))`.

    References
    ----------
    *See*, https://numpy.org/doc/stable/reference/random/parallel.html


    """

    generated_entropy = generated_entropy or [
        92156365243929466422624541055805800714117298857186959727264899187749727119124,
        45508962760932900824607908382088764294813063250106926349700153055300051503944,
        11358852481965974965852447884047438302274082458147659701772223782670581495409,
        98335771128074178116267837103565107347248838466705856121954317889296202882090,
        99169860978478959086120522268530894898455162069966492625932871292847103049882,
        87208206842095975410011581094164970201731602958127872604742955058753939512957,
        3615645999448046437740316672917313005913548649308233620056831197005377987468,
        108909094416963715978441140822183411647298834317413586830609215654532919223699,
        88096344099146385192471976829122012867254940684757663128881853302534662995332,
        63206306147411023146090085885772240748399174641427012462446714431253444120718,
    ]

    if _sseq_list_len > (_lge := len(generated_entropy)):
        _e_str_segs = (
            "This function can presently create SeedSequences for generating up to ",
            f"{_lge:,d} independent random variates. If you really need to generate ",
            f"more than {_lge:,d} seeded independent random variates, please pass a ",
            "sufficiently large list of seeds as generated_entropy. See,",
            "{}/{}.".format(
                "https://numpy.org/doc/stable/reference/random",
                "bit_generators/generated/numpy.random.SeedSequence.html",
            ),
        )
        raise ValueError("".join(_e_str_segs))

    return [SeedSequence(_s, pool_size=8) for _s in generated_entropy[:_sseq_list_len]]


class MultithreadedRNG:
    """Fill given array on demand with pseudo-random numbers as specified.

    Random number generation is multithreaded, using twice
    the number of threads as available CPU cores by default.
    If a seed sequence is provided, it is used in a thread-safe way
    to generate repeatable i.i.d. draws. All arguments are validated
    before commencing multithreaded random number generation.

    Parameters
    ----------
    _out_array
        The output array to which generated data are written.
        Its dimensions define the size of the sample.
    dist_type
        Distribution for the generated random numbers
    dist_parms
        Parameters, if any, for tailoring random number generation
    seed_sequence
        SeedSequence object for generating repeatable draws.
    nthreads
        Number of threads to spawn for random number generation.

    """

    def __init__(
        self,
        _out_array: NDArray[np.float64],
        /,
        *,
        dist_type: Literal[
            "Beta", "Dirichlet", "Gaussian", "Normal", "Random", "Uniform"
        ] = "Uniform",
        dist_parms: NDArray[np.float64] | None = DIST_PARMS_DEFAULT,
        seed_sequence: SeedSequence | None = None,
        nthreads: int = NTHREADS,
    ):
        self.thread_count = nthreads

        _seed_sequence = seed_sequence or SeedSequence(pool_size=8)
        self._random_generators = [
            prng(_t) for _t in _seed_sequence.spawn(self.thread_count)
        ]

        self.sample_sz = len(_out_array)

        if dist_type not in (_rdts := ("Beta", "Dirichlet", "Normal", "Uniform")):
            raise ValueError("Specified distribution must be one of {_rdts}")

        if not (dist_parms is None or isinstance(dist_parms, Sequence | np.ndarray)):
            raise ValueError(
                "When specified, distribution parameters must be a list, tuple or Numpy array"
            )
            if isinstance(dist_parms, Sequence):
                dist_parms = np.array(dist_parms)
            elif not dist_parms.any():
                dist_parms = None

        self.dist_type = dist_type

        if dist_parms is None or np.array_equal(dist_parms, DIST_PARMS_DEFAULT):
            match dist_type:
                case "Uniform":
                    self.dist_type = "Random"
                case "Normal":
                    self.dist_type = "Gaussian"
                case "Beta" | "Dirichlet":
                    raise ValueError(
                        f"parameter specification, {f'"{dist_parms}"'} "
                        f"is invalid for specified distribution, f{'"{dist_type}"'}."
                    )
                case _:
                    raise ValueError(
                        f"Invalid distributions specified, {f'"{dist_type}"'}."
                    )

        elif dist_type == "Dirichlet":
            if len(dist_parms) != _out_array.shape[1]:
                raise ValueError(
                    f"Insufficient shape parameters for requested Dirichlet sample "
                    f"of size, {_out_array.shape}"
                )

        elif (_lrdp := len(dist_parms)) != 2:
            raise ValueError(f"Expected 2 parameters, got, {_lrdp}")

        self.dist_parms = dist_parms

        self.values = _out_array
        self.executor = concurrent.futures.ThreadPoolExecutor(self.thread_count)

        self.step_size = (len(self.values) / self.thread_count).__ceil__()

    def fill(self) -> None:
        """Fill the provided output array with random numbers as specified."""

        def _fill(
            _rng: np.random.Generator,
            _dist_type: str,
            _dist_parms: NDArray[np.float64],
            _out: NDArray[np.float64],
            _first: int,
            _last: int,
            /,
        ) -> None:
            _sz: tuple[int, ...] = _out[_first:_last].shape
            match _dist_type:
                case "Random":
                    _rng.random(out=_out[_first:_last])
                case "Uniform":
                    _uni_l, _uni_h = _dist_parms
                    _out[_first:_last] = _rng.uniform(_uni_l, _uni_h, size=_sz)
                case "Dirichlet":
                    _out[_first:_last] = _rng.dirichlet(_dist_parms, size=_sz[:-1])
                case "Beta":
                    _shape_a, _shape_b = _dist_parms
                    _out[_first:_last] = _rng.beta(_shape_a, _shape_b, size=_sz)
                case "Normal":
                    _mu, _sigma = _dist_parms
                    _out[_first:_last] = _rng.normal(_mu, _sigma, size=_sz)
                case _:
                    _rng.standard_normal(out=_out[_first:_last])

        futures = {}
        for i in range(self.thread_count):
            _range_first = i * self.step_size
            _range_last = min(len(self.values), (i + 1) * self.step_size)
            args = (
                _fill,
                self._random_generators[i],
                self.dist_type,
                self.dist_parms,
                self.values,
                _range_first,
                _range_last,
            )
            futures[self.executor.submit(*args)] = i  # type: ignore
        concurrent.futures.wait(futures)

    def __del__(self) -> None:
        self.executor.shutdown(False)
