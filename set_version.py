#!/usr/bin/env python3

import argparse
from datetime import datetime
from pathlib import Path
from subprocess import run

import re2 as re
from semver import compare
from tomlkit import parse

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

tsn = datetime.now()


rc = run("git status".split(), check=True, capture_output=True, text=True)  # noqa: S603
if not re.search("nothing to commit, working tree clean", rc.stdout):
    raise RuntimeError(
        "Repository has uncommitted changes. Commit changes before updating package version."
    )


def _get_pkg_version(_toml_path: str = "pyproject.toml") -> str:
    _toml_dict = parse(Path(_toml_path).read_text())
    return _toml_dict["tool"]["poetry"]["version"]  # type: ignore


match args.update_level:
    case "patch":
        run(["poetry", "version", "patch"], check=True)  # noqa: S603, S607
        sem_ver = _get_pkg_version()
    case "full":
        sem_ver = f"{tsn.year}.{tsn.toordinal()}.0"
        pkg_ver = _get_pkg_version()

        if compare(sem_ver, pkg_ver) <= 0:
            raise ValueError(
                f"Package version, {pkg_ver} at or above version, {sem_ver}. Perhaps update patch-level."
            )

        run(["poetry", "version", sem_ver], check=True)  # noqa: S603, S607

run(["git", "tag", f"{sem_ver}"], check=True)  # noqa: S603, S607
run(
    ["git", "commit", "pyproject.toml", "-m", '"chore: bump version number"'],  # noqa: S603, S607
    check=True,
)
