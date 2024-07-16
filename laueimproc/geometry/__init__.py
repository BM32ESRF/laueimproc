#!/usr/bin/env python3

r"""Implement the Bragg diffraction rules.

Bases
-----

* \(\mathcal{B^c}\): The orthonormal base of the crystal
    \([\mathbf{C_1}, \mathbf{C_2}, \mathbf{C_3}]\).
* \(\mathcal{B^l}\): The orthonormal base of the lab
    \([\mathbf{L_1}, \mathbf{L_2}, \mathbf{L_3}]\) in pyfai.

Lattice parameters
------------------

* \([a, b, c, \alpha, \beta, \gamma]\): The lattice scalars parameters.
* \(\mathbf{A}\): The primitive column vectors \([\mathbf{e_1}, \mathbf{e_2}, \mathbf{e_3}]\)
    in an orthonormal base.
* \(\mathbf{B}\): The reciprocal column vectors \([\mathbf{e_1^*}, \mathbf{e_2^*}, \mathbf{e_3^*}]\)
    in an orthonormal base.

Angles
------

* \(\phi_{ij}\): The angular distance between \(u_{q_i}\) and \(u_{q_j}\), \(\phi \in [0, \pi]\).
* \(\theta\): The half deviation angle in radian in the plan \((u_f, \mathbf{L_3})\),
    \(\theta \in [0, \frac{\pi}{2}]\).
* \(\chi\): The rotation angle in radian from the vertical plan \((\mathbf{L_1}, \mathbf{L_3})\)
    to the plan \((u_f, \mathbf{L_3})\), \(\chi \in [-\pi, \pi]\).
* \([\omega_1, \omega_2, \omega_3]\): The 3 elementary rotations about an orthonormal base,
    witch consitute the rotation matrix \(\mathbf{R}\), \(
        [\omega_1, \omega_2, \omega_3]
        \in [-\pi, \pi] \times [-\frac{\pi}{2}, \frac{\pi}{2}] \times [-\pi, \pi]
    \).

Diffraction
-----------

* \(u_f\): The unit vector of the diffracted ray, often expressed in the lab base \(\mathcal{B^l}\).
* \(u_q\): The unit vector normal to the diffracting plane,
    often expressed in the lab base \(\mathcal{B^l}\).
* \(\mathbf{R}\): "La matrice de passage de \(\mathcal{B^l}\) a \(\mathcal{B^c}\)", such as
    \(\mathbf{x_{\mathcal{B^l}}} = \mathbf{R} . \mathbf{x_{\mathcal{B^c}}}\).
"""

from .bragg import (
    hkl_reciprocal_to_energy, hkl_reciprocal_to_uq, hkl_reciprocal_to_uq_energy,
    uf_to_uq, uq_to_uf,
)
from .hkl import select_hkl
from .lattice import lattice_to_primitive, primitive_to_lattice
from .metric import compute_matching_rate, compute_matching_rate_continuous, ray_cosine_dist
from .projection import detector_to_ray, ray_to_detector
from .reciprocal import primitive_to_reciprocal, reciprocal_to_primitive
from .rotation import omega_to_rot, rot_to_omega, rotate_crystal
from .thetachi import thetachi_to_uf, uf_to_thetachi


__all__ = [
    "hkl_reciprocal_to_energy", "hkl_reciprocal_to_uq", "hkl_reciprocal_to_uq_energy",
    "uf_to_uq", "uq_to_uf",
    "select_hkl",
    "lattice_to_primitive", "primitive_to_lattice",
    "compute_matching_rate", "compute_matching_rate_continuous", "ray_cosine_dist",
    "detector_to_ray", "ray_to_detector",
    "primitive_to_reciprocal", "reciprocal_to_primitive",
    "omega_to_rot", "rot_to_omega", "rotate_crystal",
    "thetachi_to_uf", "uf_to_thetachi",
]
