#!/usr/bin/env python3

"""Group some image processing tools to annalyse Laue diagrams."""

from .classes import Diagram, DiagramsDataset
from .opti import collect, DiagramManager

__all__ = ["collect", "Diagram", "DiagramsDataset", "DiagramManager"]
__author__ = "J.S. Micha, O. Robach., S. Tardif, R. Richard"
__version__ = "1.0.1"  # pep 440
