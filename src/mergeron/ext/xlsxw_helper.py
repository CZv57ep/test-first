"""
Methods for writing data from Python to fresh Excel workbooks using
the third-party package, `xlsxwriter`.

Includes a flexible system of defining cell formats.

NOTES
-----

This module is designed for producing formatted summary output. For
writing bulk data to Excel, facilities provided in third-party packages
such as `polars <https://pola.rs/>`_ likely provide better performance.

License
========

Copyright 2017-2023 S. Murthy Kambhampaty
Licese: MIT
https://mit-license.org/


"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
import xlsxwriter  # type: ignore
from aenum import Enum, extend_enum, unique  # type: ignore

from .. import VERSION  # noqa: TID252

__version__ = VERSION

Workbook = xlsxwriter.Workbook

XLBorderType: TypeAlias = Literal[
    "none",
    "thin",
    "medium",
    "dashed",
    "dotted",
    "thick",
    "double",
    "hair",
    "medium_dashed",
    "dash_dot",
    "medium_dash_dot",
    "dash_dot_dot",
    "medium_dash_dot_dot",
    "slant_dash_dot",
    True,
    False,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
]


class CFmtVal(TypedDict, total=False):
    """Keys for xlsxwriter Format objects.

    This is a partial list based on formats of interest.
    """

    font_name: str
    font_size: int
    font_color: str
    align: Literal[
        "left", "center", "right", "center_across", "top", "bottom", "vcenter"
    ]
    text_wrap: bool
    rotation: int  # integer, 0-360
    indent: int
    shrink: bool
    bold: bool
    italic: bool
    underline: Literal[
        True,
        False,
        1,
        2,
        33,
        34,
        "single",
        "double",
        "accountingSingle",
        "accountingDouble",
    ]
    font_strikeout: bool
    font_script: Literal[1, 2]

    num_format: str

    pattern: int
    fg_color: str  # html color string, no #
    bg_color: str  # html color string, no #

    hidden: bool
    locked: bool

    border: XLBorderType
    bottom: XLBorderType
    left: XLBorderType
    right: XLBorderType
    top: XLBorderType
    border_color: str  # html color string, no #
    bottom_color: str  # html color string, no #
    left_color: str  # html color string, no #
    right_color: str  # html color string, no #
    top_color: str  # html color string, no #

    diag_border: XLBorderType
    diag_border_color: str  # html color string, no #
    diag_type: Literal[
        1, 2, 3, "up", "down", "left", "right", "cross", "diagonalUp", "diagonalDown"
    ]


@unique
class CFmt(dict, Enum):  # type: ignore
    """
    Cell format enums for xlsxwriter Format objects.

    The enums defined here, or sequences of (any of) them
    and any added with :meth:`CFmt.add_new`, are
    rendered as :code:`xlsxWriter.Workbook.Format` objects
    with :meth:`CFmt.xl_fmt`.

    NOTES
    -----

    For more information about xlsxwriter cell formats,
    see, https://xlsxwriter.readthedocs.io/format.html

    """

    XL_DEFAULT: ClassVar = {"font_name": "Calibri", "font_size": 11}
    XL_DEFAULT_2003: ClassVar = {"font_name": "Arial", "font_size": 10}

    A_CTR: ClassVar = {"align": "center"}
    A_CTR_ACROSS: ClassVar = {"align": "center_across"}
    A_LEFT: ClassVar = {"align": "left"}
    A_RIGHT: ClassVar = {"align": "right"}
    V_TOP: ClassVar = {"align": "top"}
    V_BOTTOM: ClassVar = {"align": "bottom"}
    V_CTR: ClassVar = {"align": "vcenter"}

    TEXT_WRAP: ClassVar = {"text_wrap": True}
    TEXT_ROTATE: ClassVar = {"rotation": 90}
    IND_1: ClassVar = {"indent": 1}

    BOLD: ClassVar = {"bold": True}
    BOLD_ITALIC: ClassVar = {"bold": True, "italic": True}
    ITALIC: ClassVar = {"italic": True}
    ULINE: ClassVar = {"underline": "single"}
    SOUT: ClassVar = {"font_strikeout": True}
    # Useful with write_rich_text()
    SUPERSCRIPT: ClassVar = {"font_script": 1}
    SUBSCRIPT: ClassVar = {"font_script": 2}

    AREA_NUM: ClassVar = {"num_format": "0.00000000"}
    DOLLAR_NUM: ClassVar = {"num_format": "[$$-409]#,##0.00"}
    DT_NUM: ClassVar = {"num_format": "mm/dd/yyyy"}
    PCT_NUM: ClassVar = {"num_format": "##0%"}
    PCT2_NUM: ClassVar = {"num_format": "##0.00%"}
    PCT4_NUM: ClassVar = {"num_format": "##0.0000%"}
    PCT6_NUM: ClassVar = {"num_format": "##0.000000%"}
    PCT8_NUM: ClassVar = {"num_format": "##0.00000000%"}
    QTY_NUM: ClassVar = {"num_format": "#,##0.0"}

    BAR_FILL: ClassVar = {"pattern": 1, "bg_color": "dfeadf"}
    HDR_FILL: ClassVar = {"pattern": 1, "bg_color": "bfbfbf"}

    FULL_BORDER: ClassVar = {"border": 1, "border_color": "000000"}
    BOTTOM_BORDER: ClassVar = {"bottom": 1, "bottom_color": "000000"}
    LEFT_BORDER: ClassVar = {"left": 1, "left_color": "000000"}
    RIGHT_BORDER: ClassVar = {"right": 1, "right_color": "000000"}
    TOP_BORDER: ClassVar = {"top": 1, "top_color": "000000"}
    HDR_BORDER: ClassVar = TOP_BORDER | BOTTOM_BORDER

    @classmethod
    def add_new(self, _fmt_name: str, _xlsx_fmt_dict: CFmtVal, /) -> CFmt:
        """
        Add new :class:`CFmt` object to instance.

        Parameters
        ----------
        _fmt_name
            Name of new member to be added to :class:`CFmt`
        _xlsx_fmt_dict
            Any valid argument to :code:`xlsxwriter.Workbook.add_format()`, or union of
            same with one or more :class:`CFmt` objects, e.g.,
            :code:`CFmt.HDR_BORDER | CFmt.HDR_FILL`  or
            :code:`CFmt.HDR_BORDER | {"pattern": 1, "bg_color": "f2f2f2"}`

        Returns
        -------
            None

        """

        return extend_enum(self, _fmt_name, _xlsx_fmt_dict)  # type: ignore

    @classmethod
    def ensure_cell_format_spec_tuple(
        self, _cell_format: Sequence[CFmt | Sequence[CFmt]], /
    ) -> bool:
        """
        Test that a given format specification is a tuple of :class:`CFmt` enums

        Parameters
        ----------
        _cell_format
            Format specification

        Raises
        ------
        ValueError
            If format specification is not a sequence  of (sequences of)
            :class:`CFmt` enums

        Returns
        -------
            True if format specification passes, else False

        """

        for _cf in _cell_format:
            if isinstance(_cf, tuple):
                self.ensure_cell_format_spec_tuple(_cf)

            if not (isinstance(_cf, CFmt),):
                raise ValueError(
                    "Improperly specified format tuple for writing array."
                    "  Must be tuple of :class:`CFmt` enums."
                )

        return True

    @classmethod
    def xl_fmt(
        self,
        _xl_book: xlsxwriter.Workbook,
        _cell_fmt: Sequence[CFmt | Sequence[CFmt]] | CFmt | None,
        /,
    ) -> xlsxwriter.format.Format:
        """
        Return :code:`xlsxwriter` :code:`Format` object given a :class:`CFmt` enum, or tuple thereof.

        Parameters
        ----------
        _xl_book
            :code:`xlsxwriter.Workbook` object

        _cell_fmt
            :class:`CFmt` enum object, or tuple thereof

        Raises
        ------
        ValueError
            If format specification is not one of None, a :class:`CFmt` enum, or
            a :code:`xlsxwriter.format.Format` object

        Returns
        -------
            :code:`xlsxwriter` :code:`Format`  object

        """

        if isinstance(_cell_fmt, xlsxwriter.format.Format):
            return _cell_fmt
        elif _cell_fmt is None:
            return _xl_book.add_format(CFmt.XL_DEFAULT.value)

        _cell_fmt_dict: CFmtVal = {}
        if isinstance(_cell_fmt, Sequence):
            self.ensure_cell_format_spec_tuple(_cell_fmt)
            for _cf in _cell_fmt:
                _cell_fmt_dict = (
                    (_cell_fmt_dict | _cfi.value for _cfi in _cf)
                    if isinstance(_cf, Sequence)
                    else _cell_fmt_dict | _cf.value
                )
        elif isinstance(_cell_fmt, CFmt):
            _cell_fmt_dict = _cell_fmt.value
        else:
            raise ValueError("Improperly specified format specification.")

        return _xl_book.add_format(_cell_fmt_dict)


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

    if any((center_footer, left_footer, right_footer)):
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
    empty_as_blank: bool = True,
    green_bar_flag: bool = True,
    ragged_flag: bool = True,
) -> tuple[int, int]:
    """
    Write a 2-D array to a worksheet.

    The given array is required be a two-dimensional array, whether
    a nested list, nested tuple, or a 2-D numpy ndarray. The array is assumed
    to be ragged by default, i.e. not all rows are the same length, and some
    cells may contain lists, etc. For rectangular arrays, set `ragged_flag` to
    false if you wish to provide a format tuple with distinct formats for each
    column in the rectangular array.


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
        If array is rectangular but length of format tuple does not
        match row-length


    Returns
    -------
        Tuple giving address of cell at right below and after range written


    Notes
    -----

    The keyword argument cell_format may be passed a tuple of :class:`CFmt` enums,
    if, and only if, ragged_flag is False. If cell_format is a tuple, it must
    have length equal to the number of cells in each row of the passed array.
    Further, members of cell_format must each be a :class:`CFmt` enum or a
    tuple of :class:`CFmt` enums; in other words, :meth:`CFmt.ensure_cell_format_spec_tuple`
    must return True for any tuple `_c` passed as `cell_format`.

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
        if _num_rows > 1 and ragged_flag:
            raise ValueError(
                "It is not clear whether the sequence of formats applies to all cells,"
                " or to each cell respectively. Please provide a single-valued cell_format."
                " Alternatively, you can iterate over the array using scalar_to_sheet()."
            )
        elif not len(cell_format) == len(_data_table[0]):
            raise ValueError("Format tuple does not match data in length.")
        CFmt.ensure_cell_format_spec_tuple(cell_format)
        _cell_format: Sequence[CFmt | Sequence[CFmt]] = cell_format
    elif isinstance(cell_format, CFmt):
        _cell_format = (cell_format,) * len(_data_table[0])
    else:
        _cell_format = (CFmt.XL_DEFAULT,) * len(_data_table[0])

    # construct vector of xlslwrter.format.Format objects
    _wbk_formats = tuple(CFmt.xl_fmt(_xl_book, _cf) for _cf in _cell_format)
    if _num_rows > 1:
        _wbk_formats_greened = (
            tuple(
                CFmt.xl_fmt(
                    _xl_book,
                    (*_cf, CFmt.BAR_FILL)
                    if isinstance(_cf, Sequence)
                    else (_cf, CFmt.BAR_FILL),
                )
                for _cf in _cell_format
            )
            if green_bar_flag
            else _wbk_formats
        )

    for _ri, _rv in enumerate(_data_table):
        _wbk_fmt_tuple = _wbk_formats_greened if _ri % 2 else _wbk_formats
        for _ci, _cv in enumerate(_rv):
            _cf = _wbk_fmt_tuple[_ci]
            scalar_to_sheet(
                _xl_sheet,
                _row_id + _ri,
                _col_id + _ci,
                _cv,
                _cf,
                empty_as_blank=empty_as_blank,
            )

        _right_column_id = _col_id + _ci + 1 if _ci > _num_cols else _right_column_id

    return _bottom_row_id, _right_column_id


def scalar_to_sheet(
    _xl_sheet: xlsxwriter.worksheet.Worksheet,
    _cell_addr_0: str | int = "A1",
    /,
    *_s2s_args: Any,
    empty_as_blank: bool = True,
) -> None:
    """
    Write to a single cell in a worksheet.

    Parameters
    ----------
    _xl_sheet
        Worksheet object to which to write the give array

    _cell_addr_0
        First element of a cell address, which may be the entire address
        in 'A1' format or the row-part in 'Row-column' format

    _s2s_args
        Other arguments, which may be just the cell value to be written and the
        cell format, or the column-part of the 'Row-column' address along with
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

    Notes
    -----
    For more information on xlsxwriter cell-address notation, see:
    https://xlsxwriter.readthedocs.io/working_with_cell_notation.html

    """

    _cell_addr: tuple[int | str, ...] = ()
    _cell_val: Any = None
    _cell_fmt: xlsxwriter.format.Format = CFmt.XL_DEFAULT

    if isinstance(_cell_addr_0, str):
        if len(_s2s_args) not in (1, 2):
            raise ValueError("Incorrect number of arguments.")
        _cell_addr = (_cell_addr_0,)
        _cell_val = _s2s_args[0]
        _cell_fmt = _s2s_args[1] if len(_s2s_args) == 2 else None
    elif isinstance(_cell_addr_0, int):
        if len(_s2s_args) not in (2, 3) or not isinstance(_s2s_args[0], int):
            raise ValueError("Incorrect/incomplete specification for Excel cell data.")
        _cell_addr = (_cell_addr_0, _s2s_args[0])
        _cell_val = _s2s_args[1]
        _cell_fmt = _s2s_args[2] if len(_s2s_args) == 3 else None
    else:
        raise ValueError("Incorrect/incomplete specification for Excel cell data.")

    _write_args = (
        (*_cell_addr, repr(_cell_val))
        if np.ndim(_cell_val) or _cell_val in (np.inf, -np.inf, np.nan)
        else (*_cell_addr, _cell_val)
    )
    _write_args = (*_write_args, _cell_fmt) if _cell_fmt else _write_args

    if empty_as_blank and (_cell_val is None or _cell_val == ""):
        _xl_sheet.write_blank(*_write_args)
    elif (
        _cell_val is None
        or _cell_val == ""
        or isinstance(_cell_val, str)
        or np.ndim(_cell_val)
        or _cell_val in (np.inf, -np.inf, np.nan)
    ):
        _xl_sheet.write_string(*_write_args)
    else:
        _xl_sheet.write(*_write_args)
