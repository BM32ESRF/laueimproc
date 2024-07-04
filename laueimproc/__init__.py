#!/usr/bin/env python3

"""Group some image processing tools to annalyse Laue diagrams.

[main documentation](../html/index.html)
"""

from .classes import Diagram, DiagramsDataset
from .opti import collect

__all__ = ["collect", "Diagram", "DiagramsDataset"]
__author__ = "J.S. Micha, O. Robach., S. Tardif, R. Richard"
__version__ = "1.5.0"  # pep 440
