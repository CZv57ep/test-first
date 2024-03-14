"""
Routines for writing data from Python to fresh Excel workbooks using
the third-party package, `xlsxwriter`.

Includes a flexible system of defining cell formats.

"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt
import xlsxwriter  # type: ignore

from .. import _PKG_NAME  # noqa: TID252

__version__ = version(_PKG_NAME)


@enum.unique
class CFmtParent(dict[str, Any], enum.ReprEnum):  # type: ignore
    """Unique mappings defining xlsxwirter Workbook formats"""

    ...


class CFmt(CFmtParent):
    """
    Initialize cell formats for xlsxwriter.

    The mappings included here, and unions, etc. of them
    and any others added at runtime, will be rendered
    as xlsxWriter.Workbook.Format objects for writing
    cell values to formatted cells in a spreadsheet.

    See, https://xlsxwriter.readthedocs.io/format.html
    """

    XL_DEFAULT = MappingProxyType({"font_name": "Calibri", "font_size": 11})
    XL_DEFAULT_2003 = MappingProxyType({"font_name": "Arial", "font_size": 10})

    A_CTR = MappingProxyType({"align": "center"})
    A_CTR_ACROSS = MappingProxyType({"align": "center_across"})
    A_LEFT = MappingProxyType({"align": "left"})
    A_RIGHT = MappingProxyType({"align": "right"})

    BOLD = MappingProxyType({"bold": True})
    ITALIC = MappingProxyType({"italic": True})
    ULINE = MappingProxyType({"underline": True})

    TEXT_WRAP = MappingProxyType({"text_wrap": True})
    IND_1 = MappingProxyType({"indent": 1})

    DOLLAR_NUM = MappingProxyType({"num_format": "[$$-409]#,##0.00"})
    DT_NUM = MappingProxyType({"num_format": "mm/dd/yyyy"})
    QTY_NUM = MappingProxyType({"num_format": "#,##0.0"})
    PCT_NUM = MappingProxyType({"num_format": "##0.000000%"})
    AREA_NUM = MappingProxyType({"num_format": "0.00000000"})

    BAR_FILL = MappingProxyType({"pattern": 1, "bg_color": "dfeadf"})
    BOT_BORDER = MappingProxyType({"bottom": 1, "bottom_color": "000000"})
    TOP_BORDER = MappingProxyType({"top": 1, "top_color": "000000"})
    HDR_BORDER = TOP_BORDER | BOT_BORDER


def matrix_to_sheet(
    _xl_book: xlsxwriter.workbook.Workbook,
    _xl_sheet: xlsxwriter.worksheet.Worksheet,
    _data_table: npt.ArrayLike,
    _row_id: int,
    _col_id: int = 0,
    /,
    *,
    cell_format: Sequence[CFmt] | CFmt | None = None,
    green_bar_flag: bool = True,
) -> tuple[int, int]:
    """
    Write a 2-D array to a worksheet.

    The given array is required be a two-dimensional array, whether
    a nested list, nested tuple, or a 2-D numpy ndarray.

    Parameters
    ----------
    _xl_book
        Workbook object

    _xl_sheet
        Worksheet object to which to write the give array

    _data_table
        Array to be written

    _row_id
        Row number of top left corner of range to write to

    _col_id
        Column number of top left corner of range to write to

    cell_format
        Format specification for range to be written

    green_bar_flag
        Whether to highlight alternating rows as in green bar paper

    Returns
    -------
        Tuple giving address of cell (at left) below range written

    """
    _data_array: npt.NDArray[Any] = np.array(_data_table)
    del _data_table

    if not len(_data_array.shape) == 2:
        raise ValueError(
            "Array to write must be a 2-D array, but"
            f"the given array has shape, {_data_array.shape}."
        )

    # Get the array dimensions and row and column numbers for Excel
    _bottom_row_id: int = _row_id + _data_array.shape[0]
    _right_column_id: int = _col_id + _data_array.shape[1]

    if isinstance(cell_format, tuple):
        ensure_cell_format_spec_tuple(cell_format)
        if not len(cell_format) == len(_data_array[0]):
            raise ValueError("Format tuple does not match data in length.")
        _cell_format: Sequence[CFmt] = cell_format
    elif isinstance(cell_format, CFmt):
        _cell_format = (cell_format,) * len(_data_array[0])
    else:
        _cell_format = (CFmt.XL_DEFAULT,) * len(_data_array[0])

    for _cell_row in range(_row_id, _bottom_row_id):
        for _cell_col in range(_col_id, _right_column_id):
            _cell_fmt = (
                (_cell_format[_cell_col - _col_id], CFmt.BAR_FILL)
                if green_bar_flag and (_cell_row - _row_id) % 2
                else _cell_format[_cell_col - _col_id]
            )

            scalar_to_sheet(
                _xl_book,
                _xl_sheet,
                _cell_row,
                _cell_col,
                _data_array[_cell_row - _row_id, _cell_col - _col_id],
                _cell_fmt,
            )

    return _bottom_row_id, _right_column_id


def scalar_to_sheet(
    _xl_book: xlsxwriter.workbook.Workbook,
    _xl_sheet: xlsxwriter.worksheet.Worksheet,
    _cell_addr_0: str | int | float = "A1",
    /,
    *_s2s_args: Any,
) -> None:
    """
    Write to a single cell in a worksheet.

    Parameters
    ----------
    _xl_book
        Workbook object

    _xl_sheet
        Worksheet object to which to write the give array

    _cell_addr_0
        First element of a cell address, which may be the entire address
        in 'A1' format or the row-part in 'R1,C1' format

    _s2s_args
        Other arguments, which may be just the cell value to be written and the
        cell format, or the column-part of the 'R1,C1' address along with
        cell value and cell format.

    """

    if isinstance(_cell_addr_0, str):
        if len(_s2s_args) not in (1, 2):
            raise ValueError("Too many or too few arguments.")
        _cell_addr: tuple[int | str, ...] = (_cell_addr_0,)
        _cell_val: Any = _s2s_args[0]
        _cell_fmt: CFmt | Sequence[CFmt] = _s2s_args[1] if len(_s2s_args) == 2 else None  # type: ignore
    elif isinstance(_cell_addr_0, int):
        if len(_s2s_args) not in (2, 3):
            raise ValueError("Too many or too few arguments.")
        _cell_addr = (_cell_addr_0, _s2s_args[0])
        _cell_val = _s2s_args[1]
        _cell_fmt = _s2s_args[2] if len(_s2s_args) == 3 else None  # type: ignore
    else:
        raise ValueError("Incorrect specification for Excel cell data.")

    _xl_sheet.write(*_cell_addr, _cell_val, xl_fmt(_xl_book, _cell_fmt))


def xl_fmt(
    _xl_book: xlsxwriter.Workbook, _cell_fmt: Sequence[CFmt] | CFmt | None, /
) -> xlsxwriter.format.Format:
    """
    Return :code:`xlsxwriter` `Format` object given a CFmt enum, or tuple thereof.

    Parameters
    ----------
    _xl_book
        :code:`xlsxwriter.Workbook` object

    _cell_fmt
        :code:`CFmt` enum object, or tuple thereof

    Returns
    -------
        :code:`xlsxwriter` `Format`  object

    """
    _cell_fmt_dict: Mapping[str, Any] = MappingProxyType({})
    if isinstance(_cell_fmt, tuple):
        ensure_cell_format_spec_tuple(_cell_fmt)
        for _cf in _cell_fmt:
            _cell_fmt_dict = _cell_fmt_dict | _cf.value
    elif isinstance(_cell_fmt, CFmt):
        _cell_fmt_dict = _cell_fmt.value
    else:
        _cell_fmt_dict = CFmt.XL_DEFAULT.value

    return _xl_book.add_format(_cell_fmt_dict)


def ensure_cell_format_spec_tuple(_cell_formats: Sequence[CFmt], /) -> None:
    """
    Test that a given format specification is tuple of CFmt enums

    Parameters
    ----------
    _cell_formats
        Format specification

    Returns
    -------
        True if format specification passes, else False

    """

    for _cell_format in _cell_formats:
        if isinstance(_cell_format, tuple):
            ensure_cell_format_spec_tuple(_cell_format)

        if not (isinstance(_cell_format, CFmt),):
            raise ValueError("Improperly specified format tuple.")
