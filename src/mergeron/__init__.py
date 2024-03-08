from __future__ import annotations

import enum
from importlib.metadata import version
from pathlib import Path

_PKG_NAME: str = Path(__file__).parent.stem

__version__ = version(_PKG_NAME)

DATA_DIR: Path = Path.home() / _PKG_NAME
"""
Defines a subdirectory named for this package in the user's home path.

If the subdirectory doesn't exist, it is created on package invocation.
"""

if not DATA_DIR.is_dir():
    DATA_DIR.mkdir(parents=False)


@enum.unique
class RECConstants(enum.StrEnum):
    """Recapture rate - derivation methods."""

    INOUT = "inside-out"
    OUTIN = "outside-in"
    FIXED = "proportional"


@enum.unique
class UPPAggrSelector(enum.Enum):
    """
    Aggregator selection for GUPPI and diversion ratio

    """

    AVG = "average"
    CPA = "cross-product-share weighted average"
    CPD = "cross-product-share weighted distance"
    DIS = "symmetrically-weighted distance"
    MAX = "max"
    MIN = "min"
    OSA = "own-share weighted average"
    OSD = "own-share weighted distance"
