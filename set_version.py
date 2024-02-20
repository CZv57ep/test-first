#!/usr/bin/env python3

from datetime import datetime

tsn = datetime.now()
print(f"{tsn.year}.{tsn.toordinal()}.0")
