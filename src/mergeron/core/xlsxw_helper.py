"""
Methods for writing data from Python to fresh Excel workbooks using
the third-party package, `xlsxwriter`.

Includes a flexible system of defining cell formats.

NOTES
-----

This module is desinged for producing formatted summary output. For
writing bulk data to Excel, facilities provided in third-party packages
such as `polars` likely provide better performance.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import xlsxwriter  # type: ignore
from aenum import Enum, unique  # type: ignore

from .. import VERSION  # noqa: TID252

__version__ = VERSION


@unique
class CFmt(dict, Enum):  # type: ignore
    """
    Initialize cell formats for xlsxwriter.

    The mappings included here, and unions, etc. of them
    and any others added at runtime, will be rendered
    as xlsxWriter.Workbook.Format objects for writing
    cell values to formatted cells in a spreadsheet.

    NOTES
    -----

    For more information about xlsxwriter's cell formats,
    see, https://xlsxwriter.readthedocs.io/format.html
    """

    XL_DEFAULT: ClassVar = {"font_name": "Calibri", "font_size": 11}
    XL_DEFAULT_2003: ClassVar = {"font_name": "Arial", "font_size": 10}

    A_CTR: ClassVar = {"align": "center"}
    A_CTR_ACROSS: ClassVar = {"align": "center_across"}
    A_LEFT: ClassVar = {"align": "left"}
    A_RIGHT: ClassVar = {"align": "right"}

    BOLD: ClassVar = {"bold": True}
    BOLD_ITALIC: ClassVar = {"bold": True, "italic": True}
    ITALIC: ClassVar = {"italic": True}
    ULINE: ClassVar = {"underline": True}

    TEXT_WRAP: ClassVar = {"text_wrap": True}
    TEXT_ROTATE: ClassVar = {"rotation": 90}
    IND_1: ClassVar = {"indent": 1}

    DOLLAR_NUM: ClassVar = {"num_format": "[$$-409]#,##0.00"}
    DT_NUM: ClassVar = {"num_format": "mm/dd/yyyy"}
    QTY_NUM: ClassVar = {"num_format": "#,##0.0"}
    PCT_NUM: ClassVar = {"num_format": "##0%"}
    PCT2_NUM: ClassVar = {"num_format": "##0.00%"}
    PCT4_NUM: ClassVar = {"num_format": "##0.0000%"}
    PCT6_NUM: ClassVar = {"num_format": "##0.000000%"}
    PCT8_NUM: ClassVar = {"num_format": "##0.00000000%"}
    AREA_NUM: ClassVar = {"num_format": "0.00000000"}

    BAR_FILL: ClassVar = {"pattern": 1, "bg_color": "dfeadf"}
    HDR_FILL: ClassVar = {"pattern": 1, "bg_color": "999999"}

    LEFT_BORDER: ClassVar = {"left": 1, "left_color": "000000"}
    RIGHT_BORDER: ClassVar = {"right": 1, "right_color": "000000"}
    BOT_BORDER: ClassVar = {"bottom": 1, "bottom_color": "000000"}
    TOP_BORDER: ClassVar = {"top": 1, "top_color": "000000"}
    HDR_BORDER: ClassVar = TOP_BORDER | BOT_BORDER


def write_header(
    _xl_sheet: xlsxwriter.worksheet.Worksheet,
    /,
    *,
    center_header: str | None = None,
    left_header: str | None = None,
    right_header: str | None = None,
) -> None:
    """Write header text to given worksheet.

    Parameters
    ----------
    _xl_sheet
        Worksheet object
    center_header
        Text for center header
    left_header
        Text for left header
    right_header
        Text for right header

    Raises
    ------
    ValueError
        Must specify at least one header

    Returns
    -------
    None
    """
    if any((center_header, left_header, right_header)):
        _xl_sheet.set_header(
            "".join([
                f"&L{left_header}" if left_header else "",
                f"&C{center_header}" if center_header else "",
                f"&R{right_header}" if right_header else "",
            ])
        )

    else:
        raise ValueError("must specify at least one header")


def write_footer(
    _xl_sheet: xlsxwriter.worksheet.Worksheet,
    /,
    *,
    center_footer: str | None = None,
    left_footer: str | None = None,
    right_footer: str | None = None,
) -> None:
    """Write footer text to given worksheet.

    Parameters
    ----------
    _xl_sheet
        Worksheet object
    center_footer
        Text for center footer
    left_footer
        Text for left footer
    right_footer
        Text for right footer

    Raises
    ------
    ValueError
        Must specify at least one footer

    Returns
    -------
    None
    """

    if not any((center_footer, left_footer, right_footer)):
        _xl_sheet.set_footer(
            "".join([
                f"&L{left_footer}" if left_footer else "",
                f"&C{center_footer}" if center_footer else "",
                f"&R{right_footer}" if right_footer else "",
            ])
        )

    else:
        raise ValueError("must specify at least one footer")


def array_to_sheet(
    _xl_book: xlsxwriter.workbook.Workbook,
    _xl_sheet: xlsxwriter.worksheet.Worksheet,
    _data_table: Sequence[Any] | npt.NDArray[Any],
    _row_id: int,
    _col_id: int = 0,
    /,
    *,
    cell_format: Sequence[CFmt | Sequence[CFmt]] | CFmt | None = None,
    green_bar_flag: bool = True,
    ragged_flag: bool = True,
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

    ragged_flag
        Whether to write ragged array, i.e. rows not all the same length
        or not all cells are scalar-valued


    Raises
    ------
    ValueError
        If array is not two-dimensional

    ValueError
        If ragged_flag is False and array is not rectangular

    ValueError
        If array is not rectangular and cell_format is a Sequence

    ValueError
        If array is rectangular format tuple does not match data in length


    Returns
    -------
        Tuple giving address of cell at right below and after range written

    """

    if not ragged_flag:
        try:
            if np.ndim(_data_table) != 2:
                raise ValueError("Given array must be two-dimensional.")
        except ValueError as _err:
            raise ValueError(
                "Given array must be rectangular and homogenous, with scalar members."
                " Alternatively, try with ragged_flag=True."
            )
            raise _err
    elif not (
        isinstance(_data_table, Sequence | np.ndarray)
        and hasattr(_data_table[0], "__len__")
    ):
        raise ValueError("Given array must be two-dimensional array.")

    # Get the array dimensions and row and column numbers for Excel
    _num_rows = len(_data_table)
    _bottom_row_id = _row_id + _num_rows
    _num_cols = len(_data_table[0])
    _right_column_id = _col_id + _num_cols

    if isinstance(cell_format, Sequence):
        if ragged_flag:
            raise ValueError(
                "It is not clear whether the sequence of formats applies to all cells,"
                " or to each cell respectively. Please provide a single-valued cell_format."
                " Alternatively, you can iterate over the array using scalar_to_sheet()."
            )
        elif not len(cell_format) == len(_data_table[0]):
            raise ValueError("Format tuple does not match data in length.")
        ensure_cell_format_spec_tuple(cell_format)
        _cell_format: Sequence[CFmt | Sequence[CFmt]] = cell_format
    elif isinstance(cell_format, CFmt):
        _cell_format = (cell_format,) * len(_data_table[0])
    else:
        _cell_format = (CFmt.XL_DEFAULT,) * len(_data_table[0])

    # construct vector of xlslwrter.format.Format objects
    _wbk_formats = tuple(xl_fmt(_xl_book, _cf) for _cf in _cell_format)
    if _num_rows > 1:
        _wbk_formats_greened = (
            tuple(xl_fmt(_xl_book, (_cf, CFmt.BAR_FILL)) for _cf in _cell_format)
            if green_bar_flag
            else _wbk_formats
        )

    for _ri, _rv in enumerate(_data_table):
        _fmt_tuple = _wbk_formats_greened if _ri % 2 else _wbk_formats
        for _ci, _cv in enumerate(_rv):
            _cell_fmt = _fmt_tuple[_ci]
            scalar_to_sheet(
                _xl_book, _xl_sheet, _row_id + _ri, _col_id + _ci, _cv, _cell_fmt
            )

        _right_column_id = _col_id + _ci + 1 if _ci > _num_cols else _right_column_id

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

    Raises
    ------
    ValueError
        If too many or too few arguments
    ValueError
        If incorrect/incomplete specification for Excel cell data

    Returns
    -------
        None

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
        raise ValueError("Incorrect/incomplete specification for Excel cell data.")

    _xl_fmt = xl_fmt(_xl_book, _cell_fmt)
    if isinstance(_cell_val, str):
        _xl_sheet.write_string(*_cell_addr, _cell_val, _xl_fmt)
    else:
        _xl_sheet.write(
            *_cell_addr, repr(_cell_val) if np.ndim(_cell_val) else _cell_val, _xl_fmt
        )


def xl_fmt(
    _xl_book: xlsxwriter.Workbook,
    _cell_fmt: Sequence[CFmt | Sequence[CFmt]] | CFmt | None,
    /,
) -> xlsxwriter.format.Format:
    """
    Return :code:`xlsxwriter` `Format` object given a CFmt aenum, or tuple thereof.

    Parameters
    ----------
    _xl_book
        :code:`xlsxwriter.Workbook` object

    _cell_fmt
        :code:`CFmt` aenum object, or tuple thereof

    Raises
    ------
    ValueError
        If format specification is not one of None, a CFmt aenum, or
        a xlsxwriter.format.Format object

    Returns
    -------
        :code:`xlsxwriter` `Format`  object

    """

    if isinstance(_cell_fmt, xlsxwriter.format.Format):
        return _cell_fmt
    elif _cell_fmt is None:
        return _xl_book.add_format(CFmt.XL_DEFAULT.value)

    _cell_fmt_dict: Mapping[str, Any] = {}
    if isinstance(_cell_fmt, Sequence):
        ensure_cell_format_spec_tuple(_cell_fmt)
        for _cf in _cell_fmt:
            _cell_fmt_dict = _cell_fmt_dict | _cf.value
    elif isinstance(_cell_fmt, CFmt):
        _cell_fmt_dict = _cell_fmt.value
    else:
        raise ValueError("Improperly specified format specification.")

    return _xl_book.add_format(_cell_fmt_dict)


def ensure_cell_format_spec_tuple(
    _cell_formats: Sequence[CFmt | Sequence[CFmt]], /
) -> None:
    """
    Test that a given format specification is tuple of CFmt enums

    Parameters
    ----------
    _cell_formats
        Format specification

    Raises
    ------
    ValueError
        If format specification is not tuple of CFmt enums

    Returns
    -------
        True if format specification passes, else False

    """

    for _cell_format in _cell_formats:
        if isinstance(_cell_format, tuple):
            ensure_cell_format_spec_tuple(_cell_format)

        if not (isinstance(_cell_format, CFmt),):
            raise ValueError(
                "Improperly specified format tuple for writing array."
                "  Must be tuple of CFmt enums."
            )
