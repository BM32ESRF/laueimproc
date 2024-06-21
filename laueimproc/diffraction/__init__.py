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
    \([\mathbf{X_1}, \mathbf{X_2}, \mathbf{X_3}]\) in pyfai.

Lattice parameters
------------------

https://en.wikipedia.org/wiki/Lattice_constant#/media/File:UnitCell.png

* \([a, b, c, \alpha, \beta, \gamma]\): The latice parameters.
* \(\mathbf{A}\): The column vectors \([\mathbf{e_1}, \mathbf{e_2}, \mathbf{e_3}]\)
    in the base \(\mathcal{B^c}\).
* \(\mathbf{B}\): The column vectors \([\mathbf{e_1^*}, \mathbf{e_2^*}, \mathbf{e_3^*}]\)
    in the base \(\mathcal{B^c}\).
"""

from .lattice import lattice_to_primitive, primitive_to_lattice


__all__ = [
    lattice_to_primitive, primitive_to_lattice
]
