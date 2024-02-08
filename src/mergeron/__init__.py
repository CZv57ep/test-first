from importlib import metadata
from pathlib import Path

_PKG_NAME: str = Path(__file__).parent.stem

__version__ = metadata.version(_PKG_NAME)

DATA_DIR: Path = Path.home() / _PKG_NAME
"""
Defines a subdirectory named for this package in the user's home path.

If the subdirectory doesn't exist, it is created on package invocation.
"""

if not DATA_DIR.is_dir():
    DATA_DIR.mkdir(parents=False)
