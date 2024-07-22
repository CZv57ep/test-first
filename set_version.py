#!/usr/bin/env python3

import argparse
from pathlib import Path
from subprocess import PIPE, STDOUT, run

import pendulum
import re2 as re  # type: ignore
from semver import compare

# Set up the argument parser
parser = argparse.ArgumentParser(
    description="Updates package version number, and commits and tags repository. User must specify `full` or `patch` level update."
)
parser.add_argument(
    "update_level",
    type=str,
    choices=["full", "patch"],
    help="Whether `full` or `patch` level version update.",
)

args = parser.parse_args()

tsn = pendulum.today()

rc = run(["git", "status", "-uno"], check=True, stdout=PIPE, stderr=STDOUT, text=True)  # noqa: S603, S607
if not re.search("nothing to commit", rc.stdout):
    raise RuntimeError(
        "Repository has uncommitted changes. Commit changes before updating package version."
    )

# Update README.rst from the docs version
strip_sphinx_pat = re.compile(r":(attr|class|meth|mod):")
Path("./README.rst").write_text(strip_sphinx_pat.sub("", Path("./docs/source/README.rst").read_text()))

def _get_pkg_version() -> str:
    return run(  # noqa: S603
        ["poetry", "version", "-s"],  # noqa: S607
        stdout=PIPE,
        text=True,
        check=True,
    ).stdout.strip()


pkg_ver = _get_pkg_version()
sem_ver = f"{tsn.year}.{tsn.toordinal()}.0"

# Update pyproject.toml
match args.update_level:
    case "patch":
        run(["poetry", "version", "patch"], check=True)  # noqa: S607
        sem_ver = _get_pkg_version()
    case "full":
        if compare(sem_ver, pkg_ver) <= 0:
            raise ValueError(
                f"Package version, {pkg_ver} at or above version, {sem_ver}. Perhaps update patch-level."
            )

        run(["poetry", "version", sem_ver], check=True)  # noqa: S603, S607

# Update package's main __init__.py
pkg_init_path = (_p := Path(__file__).parent) / "src" / _p.name / "__init__.py"
pkg_init_path.write_text(
    re.sub(
        rf'(?m)^VERSION = "{pkg_ver}"$',
        f'VERSION = "{sem_ver}"',
        pkg_init_path.read_text(),
    )
)

run(["git", "tag", f"{sem_ver}"], check=True)  # noqa: S603, S607
run(  # noqa: S603
    [  # noqa: S607
        "git",
        "commit",
        "pyproject.toml",
        f"{pkg_init_path}",
        "-m",
        '"chore: update version"',
    ],
    check=True,
)
