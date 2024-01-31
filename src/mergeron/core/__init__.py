from importlib import metadata
from pathlib import Path

__version__ = metadata.version(Path(__file__).parents[1].stem)
