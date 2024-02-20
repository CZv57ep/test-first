#!/usr/bin/env python3

from datetime import datetime
from subprocess import run

tsn = datetime.now()
sem_ver = f"{tsn.year}.{tsn.toordinal()}.0"

run(["poetry", "version", sem_ver], check=True)  # noqa: S603, S607
run(["git", "tag", sem_ver], check=True)  # noqa: S603, S607
