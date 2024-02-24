#!/usr/bin/env python3

from datetime import datetime
from subprocess import run

tsn = datetime.now()
sem_ver = f"{tsn.year}.{tsn.toordinal()}.0"

rc = run("git status".split(), check=True)  # noqa: S603
if not rc.returncode:
    run(["poetry", "version", sem_ver], check=True)  # noqa: S603, S607
    run(["git", "tag", sem_ver], check=True)  # noqa: S603, S607
else:
    raise RuntimeError(
        "Repository has uncommitted changes. Commit changes before updating package version."
        )
