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
from laueimproc import __author__, __version__


# -- Project information -----------------------------------------------------
project = "laueimproc"
author = __author__
version = __version__
release = version
now = datetime.datetime.now()
today = f"{now.year}-{now.month:02}-{now.day:02} {now.hour:02}H{now.minute:02}"
copyright = f"2024-{now.year}, {author}"
source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]
html_theme = "sphinx_rtd_theme"
pygments_style = "sphinx"
extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks", # in pillow conf
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx", # in pillow conf
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode", # in pillow conf
    "sphinx_copybutton", # in pillow conf
    "sphinx_inline_tabs", # in pillow conf
    "sphinx_rtd_theme",
    "sphinxext.opengraph", # in pillow conf
]
todo_include_todos = True
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}


# -- Pdoc3 auto generation----------------------------------------------------
import subprocess
subprocess.run(  # build with pdoc3
    [
        "python", "-m", "pdoc",
        "--html", "--force", "--config", "latex_math=True",
        "--output-dir", str(pathlib.Path(__file__).resolve().parent / "build" / "html"),
        str(pathlib.Path(__file__).resolve().parent.parent / "laueimproc"),
    ],
    shell=False,
    check=True,
)
