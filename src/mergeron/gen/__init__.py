from importlib.metadata import version

from .. import _PKG_NAME  # noqa: TID252

__version__ = version(_PKG_NAME)
