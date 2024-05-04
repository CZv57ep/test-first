"""

The mergeron project documentation follows numpydoc style, with
the exception that the return type is given in the function
signature and not included under the heading, "Returns". This
follows from the fact that mergeron is a typed package, so we avoid
redundant type annotations except when defining class attributes,
type annotations of which aren't included in documentation by Sphinx
and extensions.

"""

import sys
from pathlib import Path

import jinja2
import mergeron
import semver

version_dict = semver.parse(mergeron.__version__)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mergeron"
copyright = "2017--2024, S. Murthy Kambhampaty"
author = "S. Murthy Kambhampaty"
version = "{major}.{minor}".format(**version_dict)
release = "{major}.{minor}.{patch}".format(**version_dict)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "autoapi.extension",
    "sphinx_immaterial",
]

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

autoapi_python_class_content = "class"
autoapi_add_toctree_entry = False
autoapi_member_order = "source"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]


autoapi_template_dir = "_autoapi_templates"
# autoapi_keep_files = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://jbms.github.io/sphinx-immaterial/customization.html
# https://github.com/jbms/sphinx-immaterial/issues/25
html_theme = "sphinx_immaterial"
html_theme_options = {
    "navigation_with_keys": False,
    "features": ["navigation.top", "navigation.tracking", "toc.follow"],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "toggle": {
                "icon": "material/toggle-switch-off-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "toggle": {
                "icon": "material/toggle-switch",
                "name": "Switch to light mode",
            },
        },
    ],
}

html_logo = "mergeron-logo.svg"
html_static_path = ["_static"]
html_title = f"{project} {release}"
html_short_title = f"{project}"

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, f'{Path.resolve(Path("../../src"))}')
autoapi_dirs = ["../../src/"]
