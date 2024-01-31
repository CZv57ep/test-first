import re
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from types import MappingProxyType

import fitz
import msgpack

__version__ = metadata.version(Path(__file__).parents[1].stem)


data_dir = Path.home() / Path(__file__).parents[1].stem / "FTCData"
if not data_dir.is_dir():
    data_dir.mkdir(parents=True)
invdata_dump_path = data_dir.parent / "billmansalop_data_tables.msgpack"

table_no_re = re.compile(r"Table A(\d)\. .*")

_invdata_path = Path(
    R"M:\SMK-PersonalFolders\AuthorshipProjects",
    R"GUPPISafeHarborTAB\LiteratureCited",
    R"BillmanSalop_2022_MergerEnforcementStatistics_2001_2020_SSRN-id4274304.pdf",
)


def parse_invdata(_invdata_pdf_path: Path) -> Mapping:
    _invdata_fitz = fitz.open(_invdata_pdf_path)

    _data_period = "".join("2001-2020")

    _table_data_dict = {}
    _pgdata_list = []
    for _invdata_pageno in range(28, 45):
        _invdata_page = _invdata_fitz[_invdata_pageno]
        _pgdata_row = ""
        for _pgblk in _invdata_page.get_text("blocks", sort=False):
            _pgdata = ", ".join(_f.strip() for _f in _pgblk[-3].strip().split("\n"))

            if _pgdata.startswith("Table A"):
                _tblno = _pgdata.split(".")[0]
                _table_data_dict |= {_tblno: []}
                _pgdata_list = _table_data_dict[_tblno]
            elif (
                _pgdata.startswith((
                    "Electronic copy available at:",
                    "<image:",
                    "Parties, Agency",
                ))
                or re.match(
                    r"(Filed|Resolved|Settled), (Resolved, )?Consummated", _pgdata
                )
                or re.match(r"Complaint|Outcome|Filed", _pgdata)
                or re.match(r"\d+|, +|^$", _pgdata)
                or hasattr(_pgdata, "image")
            ):
                continue
            elif not re.search(r"DOJ|FTC", _pgdata):
                _pgdata_row = "{}, ".format(
                    " ".join(
                        _f.strip() for _f in _pgdata.rsplit(",", maxsplit=1)
                    ).strip()
                )
                if _pgdata_row.startswith("H.J. Heinz"):
                    _pgdata_row = _pgdata_row.replace(R"Beech- Nut", "Beech-Nut")
            else:
                _pgdata_row += _pgdata  # .replace(R"Beech-\nNut", "Beech-Nut")
                _pgdata_list += [_pgdata_row]
                _pgdata_row = ""

    _table_data_dict = MappingProxyType(_table_data_dict)
    _ = invdata_dump_path.write_bytes(msgpack.packb(_table_data_dict))

    return _table_data_dict


if __name__ == "__main__":
    table_data_dict = parse_invdata(_invdata_path)
    for tblno in table_data_dict:
        print(tblno, len(table_data_dict[tblno]), sep=", ")
        for tblrow in table_data_dict[tblno]:
            print(tblrow)
        print()
        print()
