#!/usr/bin/env python3

"""Combines atomic functions to provide an overview."""

import networkx as nx

from .hkl import select_hkl
from .lattice import lattice_to_primitive, primitive_to_lattice
from .reciprocal import primitive_to_reciprocal, reciprocal_to_primitive
from .rotation import omega_to_rot, rot_to_omega, rotate_crystal
from .bragg import hkl_reciprocal_to_uq_energy, uf_to_uq, uq_to_uf


LINKS = nx.DiGraph(name="Geometry Linker Graph")
LINKS.add_nodes_from(
    [
        "lattice", "primitive_bc", "reciprocal_bc",
        "angle", "rot",
        "primitive_bl", "reciprocal_bl",
        "e_max", "e_min",
        "hkl_all", "hkl", "energy", "u_q", "u_f",
        "poni", "point",
    ],
    type="data",
    value=None,
)
LINKS.add_nodes_from(
    [
        ("lattice_to_primitive_bc", {"value": lattice_to_primitive}),
        ("primitive_to_lattice", {"value": primitive_to_lattice}),
        ("primitive_to_reciprocal", {"value": primitive_to_reciprocal}),
        ("reciprocal_to_primitive", {"value": reciprocal_to_primitive}),
        ("omega_to_rot", {"value": omega_to_rot}),
        ("rot_to_omega", {"value": rot_to_omega}),
        ("rotate_crystal", {"value": rotate_crystal}),
        ("select_hkl", {"value": select_hkl}),
        ("hkl_reciprocal_to_uq_energy", {"value": hkl_reciprocal_to_uq_energy}),
        ("uq_to_uf", {"value": uq_to_uf}),
        ("uf_to_uq", {"value": uf_to_uq}),
    ],
    type="func",
)
LINKS.add_edges_from([
    ("lattice", "lattice_to_primitive_bc"), ("lattice_to_primitive_bc", "primitive_bc"),
])
