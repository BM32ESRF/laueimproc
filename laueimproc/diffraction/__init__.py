#!/usr/bin/env python3

r"""Implement the Bragg diffraction rules.

https://www.silx.org/doc/pyFAI/latest/geometry.html#detector-position

Bases
-----

* \(\mathcal{B^c}\): The orthonormal base of the cristal
    \([\mathbf{C_1}, \mathbf{C_2}, \mathbf{C_3}]\).
* \(\mathcal{B^s}\): The orthonormal base of the sample
    \([\mathbf{S_1}, \mathbf{S_2}, \mathbf{S_3}]\).
* \(\mathcal{B^l}\): The orthonormal base of the lab
    \([\mathbf{L_1}, \mathbf{L_2}, \mathbf{L_3}]\) in pyfai.

Lattice parameters
------------------

https://en.wikipedia.org/wiki/Lattice_constant#/media/File:UnitCell.png

* \([a, b, c, \alpha, \beta, \gamma]\): The latice parameters.
* \(\mathbf{A}\): The primitive column vectors \([\mathbf{e_1}, \mathbf{e_2}, \mathbf{e_3}]\)
    in an orthonormal base.
* \(\mathbf{B}\): The reciprocal column vectors \([\mathbf{e_1^*}, \mathbf{e_2^*}, \mathbf{e_3^*}]\)
    in an orthonormal base.
"""

from .bragg import (
    reciprocal_hkl_to_energy, reciprocal_hkl_to_uq, reciprocal_hkl_to_uq_energy,
    uf_to_uq, uq_to_uf,
)
from .lattice import lattice_to_primitive, primitive_to_lattice
from .reciprocal import primitive_to_reciprocal, reciprocal_to_primitive


__all__ = [
    "reciprocal_hkl_to_energy", "reciprocal_hkl_to_uq", "reciprocal_hkl_to_uq_energy",
    "uf_to_uq", "uq_to_uf",
    "lattice_to_primitive", "primitive_to_lattice",
    "primitive_to_reciprocal", "reciprocal_to_primitive",
]
