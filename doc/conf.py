#!/usr/bin/env python3

"""
** Configuration for generated the Sphinx documenation. **
----------------------------------------------------------
"""

# -- Path setup --------------------------------------------------------------
import datetime
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from laueimproc import __version__

# -- Project information -----------------------------------------------------
project = "laueimproc"
author = "J.S. Micha, O. Robach., S. Tardif, R. Richard"
version = __version__
release = version
now = datetime.now()
today = f"{now.year}-{now.month:02}-{now.day:02} {now.hour:02}H{now.minute:02}"
copyright = f"2024-{now.year}, {author}"
source_suffix = ".rst"
master_doc = "index"
language = None
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]
html_theme = "sphinx_rtd_theme"
pygments_style = "sphinx"
extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx_tabs.tabs",
]
todo_include_todos = True
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}
