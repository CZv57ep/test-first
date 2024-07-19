from __future__ import annotations

import enum
from pathlib import Path
from typing import Any

import numpy as np
import pendulum  # type: ignore
from icecream import argumentToString, ic, install  # type: ignore
from numpy.typing import NDArray

_PKG_NAME: str = Path(__file__).parent.stem

VERSION = "2024.739079.13"

__version__ = VERSION

DATA_DIR: Path = Path.home() / _PKG_NAME
"""
Defines a subdirectory named for this package in the user's home path.

If the subdirectory doesn't exist, it is created on package invocation.
"""
if not DATA_DIR.is_dir():
    DATA_DIR.mkdir(parents=False)


np.set_printoptions(precision=18)


def _timestamper() -> str:
    return f"{pendulum.now().strftime("%F %T.%f")} |>  "


@argumentToString.register(np.ndarray)  # type: ignore
def _(_obj: NDArray[Any]) -> str:
    return f"ndarray, shape={_obj.shape}, dtype={_obj.dtype}"


ic.configureOutput(prefix=_timestamper, includeContext=True)
install()


@enum.unique
class RECConstants(enum.StrEnum):
    """Recapture rate - derivation methods."""

    INOUT = "inside-out"
    OUTIN = "outside-in"
    FIXED = "proportional"


@enum.unique
class UPPAggrSelector(enum.StrEnum):
    """
    Aggregator selection for GUPPI and diversion ratio

    """

    AVG = "average"
    CPA = "cross-product-share weighted average"
    CPD = "cross-product-share weighted distance"
    CPG = "cross-product-share weighted geometric mean"
    DIS = "symmetrically-weighted distance"
    GMN = "geometric mean"
    MAX = "max"
    MIN = "min"
    OSA = "own-share weighted average"
    OSD = "own-share weighted distance"
    OSG = "own-share weighted geometric mean"
