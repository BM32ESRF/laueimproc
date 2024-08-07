#!/usr/bin/env python3

"""Combines atomic functions to provide an overview."""

import numbers
import warnings

import networkx as nx
import torch

from .bragg import hkl_reciprocal_to_uq_energy, uf_to_uq, uq_to_uf
from .hkl import select_hkl
from .lattice import lattice_to_primitive, primitive_to_lattice
from .projection import detector_to_ray, ray_to_detector
from .reciprocal import primitive_to_reciprocal, reciprocal_to_primitive
from .rotation import omega_to_rot, rot_to_omega, rotate_crystal
from .thetachi import thetachi_to_uf, uf_to_thetachi


GRAPH = nx.DiGraph(name="Geometry Linker Graph")
GRAPH.add_nodes_from(  # data
    [
        "lattice", "primitive_bc", "reciprocal_bc",
        "omega", "rot",
        "primitive_bl", "reciprocal_bl",
        "e_max", "e_min",
        "hkl_all", "hkl", "energy", "u_q", "u_f",
        "poni", "point", "dist",
        "gnomonic", "poni_gnom",
        "thetachi",
    ],
    type="data",
    value=None,
    style="filled",
    fillcolor="coral",

)
GRAPH.add_nodes_from(  # functions
    [
        ("lattice_to_primitive_bc", {
            "value": lattice_to_primitive, "label": "lattice_to_primitive"
        }),
        ("primitive_bc_to_lattice", {
            "value": primitive_to_lattice, "label": "primitive_to_lattice"
        }),
        ("primitive_bl_to_lattice", {
            "value": primitive_to_lattice, "label": "primitive_to_lattice"
        }),
        ("primitive_bc_to_reciprocal_bc", {
            "value": primitive_to_reciprocal, "label": "primitive_to_reciprocal"
        }),
        ("primitive_bl_to_reciprocal_bl", {
            "value": primitive_to_reciprocal, "label": "primitive_to_reciprocal"
        }),
        ("reciprocal_bc_to_primitive_bc", {
            "value": reciprocal_to_primitive, "label": "reciprocal_to_primitive"
        }),
        ("reciprocal_bl_to_primitive_bl", {
            "value": reciprocal_to_primitive, "label": "reciprocal_to_primitive"
        }),
        ("omega_to_rot", {
            "value": omega_to_rot, "label": "omega_to_rot"
        }),
        ("rot_to_omega", {
            "value": rot_to_omega, "label": "rot_to_omega"
        }),
        ("rotate_primitive", {
            "value": rotate_crystal, "label": "rotate_primitive"
        }),
        ("rotate_reciprocal", {
            "value": rotate_crystal, "label": "rotate_reciprocal"
        }),
        ("select_hkl", {
            "value": select_hkl, "label": "select_hkl"
        }),
        ("hkl_reciprocal_to_uq_energy", {
            "value": hkl_reciprocal_to_uq_energy, "lsbel": "hkl_reciprocal_to_uq_energy"
        }),
        ("uq_to_uf", {
            "value": uq_to_uf, "label": "uq_to_uf"
        }),
        ("uf_to_uq", {
            "value": uf_to_uq, "label": "uf_to_uq"
        }),
        ("uf_to_detector", {
            "value": ray_to_detector, "label": "ray_to_detector"
        }),
        ("uq_to_detector", {
            "value": ray_to_detector, "label": "ray_to_detector"
        }),
        ("detector_to_uf", {
            "value": detector_to_ray, "label": "detector_to_ray"
        }),
        ("detector_to_uq", {
            "value": detector_to_ray, "label": "detector_to_ray"
        }),
        ("uf_to_thetachi", {
            "value": uf_to_thetachi, "label": "uf_to_thetachi"
        }),
        ("thetachi_to_uf", {
            "value": thetachi_to_uf, "label": "thetachi_to_uf"
        }),
    ],
    type="func",
    shape="rectangle",

)
GRAPH.add_edges_from([
    ("lattice", "lattice_to_primitive_bc"), ("lattice_to_primitive_bc", "primitive_bc"),
    ("primitive_bc", "primitive_bc_to_lattice"), ("primitive_bc_to_lattice", "lattice"),
    ("primitive_bl", "primitive_bl_to_lattice"), ("primitive_bl_to_lattice", "lattice"),
    ("omega", "omega_to_rot"), ("omega_to_rot", "rot"),
    ("rot", "rot_to_omega"), ("rot_to_omega", "omega"),
    ("primitive_bc", "rotate_primitive"), ("rot", "rotate_primitive"),
    ("rotate_primitive", "primitive_bl"),
    ("reciprocal_bc", "rotate_reciprocal"), ("rot", "rotate_reciprocal"),
    ("rotate_reciprocal", "reciprocal_bl"),
    ("primitive_bc", "primitive_bc_to_reciprocal_bc"),
    ("primitive_bc_to_reciprocal_bc", "reciprocal_bc"),
    ("primitive_bl", "primitive_bl_to_reciprocal_bl"),
    ("primitive_bl_to_reciprocal_bl", "reciprocal_bl"),
    ("reciprocal_bc", "reciprocal_bc_to_primitive_bc"),
    ("reciprocal_bc_to_primitive_bc", "primitive_bc"),
    ("reciprocal_bl", "reciprocal_bl_to_primitive_bl"),
    ("reciprocal_bl_to_primitive_bl", "primitive_bl"),
    ("e_max", "select_hkl"), ("reciprocal_bl", "select_hkl"), ("select_hkl", "hkl_all"),
    ("reciprocal_bl", "hkl_reciprocal_to_uq_energy"),
    ("hkl_all", "hkl_reciprocal_to_uq_energy"),
    ("e_min", "hkl_reciprocal_to_uq_energy"),
    ("hkl_reciprocal_to_uq_energy", "u_q"),
    ("hkl_reciprocal_to_uq_energy", "hkl"),
    ("hkl_reciprocal_to_uq_energy", "energy"),
    ("u_q", "uq_to_uf"), ("uq_to_uf", "u_f"),
    ("u_f", "uf_to_uq"), ("uf_to_uq", "u_q"),
    ("u_f", "uf_to_detector"), ("poni", "uf_to_detector"),
    ("uf_to_detector", "point"), ("uf_to_detector", "dist"),
    ("point", "detector_to_uf"), ("poni", "detector_to_uf"),
    ("detector_to_uf", "u_f"),
    ("u_q", "uq_to_detector"), ("poni_gnom", "uq_to_detector"),
    ("uq_to_detector", "gnomonic"), ("uq_to_detector", "dist"),
    ("gnomonic", "detector_to_uq"), ("poni_gnom", "detector_to_uq"),
    ("detector_to_uq", "u_q"),
    ("u_f", "uf_to_thetachi"), ("uf_to_thetachi", "thetachi"),
    ("thetachi", "thetachi_to_uf"), ("thetachi_to_uf", "u_f"),
])


class Geometry(torch.nn.Module):
    r"""General simulation class.

    Attributes
    ----------
    cam_size : None | tuple[float, float]
        The size of the detector in m.
    e_min : float
        Low energy limit in J (read and write).
    e_max : float
        High energy limit in J (read and write).
    lattice : None | torch.Tensor
        The lattice parameters of shape (..., 6) (read and write).
    omega : None | torch.Tensor
        The rotation angles of the different grains of shape (..., 3) (read and write).
    poni : torch.Tensor, optional
        The camera calibration parameters of shape (..., 6) (read and write).
    primitive_bc : None | torch.Tensor
        Matrix \(\mathbf{A}\) of shape (..., 3, 3) in the crystal base \(\mathcal{B^c}\) (readonly).
    reciprocal_bc : None | torch.Tensor
        Matrix \(\mathbf{B}\) of shape (..., 3, 3) in the crystal base \(\mathcal{B^c}\) (readonly).
    reciprocal_bl : None | torch.Tensor
        Matrix \(\mathbf{B}\) of shape (..., 3, 3) in the lab base \(\mathcal{B^l}\) (readonly).
    rot : None | torch.Tensor
        The rotation matrix of shape (..., 3, 3) (readonly).
    u_f : None | torch.Tensor
        The ravel \(u_f\) of shape (n, 3) (readonly).
    u_q : None | torch.Tensor
        The ravel \(u_q\) of shape (n, 3) (readonly).

    Todo
    ----
    This class should be recoded in a more general way, based on a graph.
    The current implementation is crude, non-modular and partial.
    """

    def __init__(self, **kwargs):
        """Initialise the model.

        Parameters
        ----------
        cam_size : tuple[float, float]
            The size of the detector in m.
        e_min, e_max : float
            The light energy band in J.
        lattice : torch.Tensor, optional
            The lattice parameters of the grains to be simulated of shape (..., 6).
        omega : torch.Tensor, optional
            The rotation angles of the different grains of shape (..., 3).
        poni : torch.Tensor, optional
            The camera calibration parameters of shape (..., 6).
        """
        assert set(kwargs).issubset({"cam_size", "e_min", "e_max", "lattice", "poni", "omega"}), \
            sorted(kwargs)

        super().__init__()

        self._cam_size = None
        self._hkl_args = {}
        self._lattice = None
        self._omega = None
        self._poni = None
        self._cache = {}

        self._cam_size = self._select_cam_size(kwargs)
        self._hkl_args.update(self._select_hkl_args(kwargs))  # e_min, e_max
        self._lattice = self._select_lattice(kwargs)
        if self._lattice is not None:
            self._lattice = torch.nn.Parameter(self._lattice)
        self._omega = self._select_omega(kwargs)
        if self._omega is not None:
            self._omega = torch.nn.Parameter(self._omega)
        self._poni = self._select_poni(kwargs)
        if self._poni is not None:
            self._poni = torch.nn.Parameter(self._poni)

    def _select_cam_size(self, kwargs: dict) -> None | tuple[float, float]:
        """Recover and checks the parameters `cam_size`."""
        if (cam_size := kwargs.get("cam_size", None)) is None:
            return self._cam_size
        if self._cam_size is not None:
            warnings.warn(
                "the `cam_size` parameter has already been supplied at initialisation",
                RuntimeWarning,
            )
        assert hasattr(cam_size, "__iter__"), cam_size.__class__.__name__
        cam_size = list(cam_size)
        assert len(cam_size) == 2, len(cam_size)
        assert isinstance(cam_size[0], numbers.Real) and isinstance(cam_size[1], numbers.Real)
        assert cam_size[0] > 0 and cam_size[1] > 0, cam_size
        return (float(cam_size[0]), float(cam_size[1]))

    def _select_hkl_args(self, kwargs: dict) -> None | dict:
        """Recover and checks the parameters e_min and e_max."""
        hkl_args = {}
        if (e_min := kwargs.get("e_min", None)) is not None:
            assert isinstance(e_min, numbers.Real), e_min.__class__.__name__
            assert e_min >= 0.0, e_min
            hkl_args["e_min"] = float(e_min)
            if "e_min" in self._hkl_args:
                warnings.warn(
                    "the `e_min` parameter has already been supplied at initialisation",
                    RuntimeWarning,
                )
        elif (e_min := self._hkl_args.get("e_min", None)) is not None:
            hkl_args["e_min"] = e_min
        if (e_max := kwargs.get("e_max", None)) is not None:
            assert isinstance(e_max, numbers.Real), e_max.__class__.__name__
            assert e_max >= hkl_args.get("e_min", 0.0), e_max
            hkl_args["e_max"] = float(e_max)
            if "e_max" in self._hkl_args:
                warnings.warn(
                    "the `e_max` parameter has already been supplied at initialisation",
                    RuntimeWarning,
                )
        elif (e_max := self._hkl_args.get("e_max", None)) is not None:
            hkl_args["e_max"] = e_max
        return hkl_args

    def _select_lattice(self, kwargs: dict) -> None | torch.Tensor:
        """Recover and checks the parameter `lattice`."""
        if (lattice := kwargs.get("lattice", None)) is None:
            return self._lattice
        if self._lattice is not None:
            warnings.warn(
                "the `lattice` parameter has already been supplied at initialisation",
                RuntimeWarning,
            )
        assert isinstance(lattice, torch.Tensor), lattice.__class__.__name__
        assert lattice.shape[-1:] == (6,), lattice.shape
        return lattice

    def _select_omega(self, kwargs: dict) -> None | torch.Tensor:
        """Recover and checks the parameter `omega`."""
        if (omega := kwargs.get("omega", None)) is None:
            return self._omega
        if self._omega is not None:
            warnings.warn(
                "the `omega` parameter has already been supplied at initialisation",
                RuntimeWarning,
            )
        assert isinstance(omega, torch.Tensor), omega.__class__.__name__
        assert omega.shape[-1:] == (3,), omega.shape
        return omega

    def _select_poni(self, kwargs: dict) -> None | torch.Tensor:
        """Recover and checks the parameter `poni`."""
        if (poni := kwargs.get("poni", None)) is None:
            return self._poni
        if self._poni is not None:
            warnings.warn(
                "the `poni` parameter has already been supplied at initialisation",
                RuntimeWarning,
            )
        assert isinstance(poni, torch.Tensor), poni.__class__.__name__
        assert poni.shape[-1:] == (6,), poni.shape
        return poni

    def _compute_candidate_hkl(self, **kwargs) -> torch.Tensor:
        """Get all hkl indices that could potentially diffract."""
        reciprocal_bc = self.compute_reciprocal_bc(**kwargs)
        if (e_max := self._select_hkl_args(kwargs).get("e_max", None)) is None:
            raise AttributeError("`e_max` parameter has to be supplied")
        signature = ("candidate_hkl", reciprocal_bc.data_ptr(), e_max)
        if (hkl := self._cache.get(signature, None)) is None:
            hkl = self._cache[signature] = select_hkl(reciprocal=reciprocal_bc, e_max=e_max)
        return hkl

    def compute_cam(
        self, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the position of the points on the camera.

        Returns
        -------
        point : torch.Tensor
            The ravel points of shape (n, 2).
        energy : torch.Tensor
            The energy in J of each spot of shape (n,).
        hkl : torch.Tensor
            The full hkl list of shape (m, 3).
        unravel : torch.Tensor
            The int64 indice of each batch dimension of shape (n, o).
            It means that the ith elements comme from the batch unravel[i].
            The dimensions are the batch dimension of ``self.reciprocal_bl + poni``.

        Examples
        --------
        >>> import torch
        >>> from laueimproc.geometry.link import Geometry
        >>> e_max = 25e3 * 1.60e-19  # 25 keV
        >>> lattice = torch.tensor([3.6e-10, 3.6e-10, 3.6e-10, torch.pi/2, torch.pi/2, torch.pi/2])
        >>> poni = torch.tensor([0.07, 73.4e-3, 73.4e-3, 0.0, -torch.pi/2, 0.0])
        >>> omega = torch.zeros(3)
        >>> model = Geometry(e_max=e_max, lattice=lattice, omega=omega, poni=poni)
        >>> point, energy, hkl, _ = model.compute_cam()
        >>>
        """
        u_f, energy, hkl, unravel = self.compute_uf(**kwargs)  # (n,)
        if (poni := self._select_poni(kwargs)) is None:  # (...,)
            raise AttributeError("`poni` parameter has to be supplied")

        poni_b = poni.shape[:-1]
        point, dist = ray_to_detector(u_f, poni, cartesian_product=True)  # (n, ...)
        cond = dist > 0  # (n, ...)
        if (cam_size := self._select_cam_size(kwargs)) is not None:
            cond &= (
                (point[..., 0] >= 0) & (point[..., 1] >= 0)
                & (point[..., 0] <= cam_size[0]) & (point[..., 1] <= cam_size[1])
            )
        energy = energy[:, *((None,)*len(poni_b))].expand(-1, *poni_b)  # (n, ...)
        hkl = hkl[:, *((None,)*len(poni_b)), :].expand(-1, *poni_b, -1)  # (n, ..., 3)

        point = point[cond]  # (n', 2)
        energy = energy[cond]  # (n',)
        hkl = hkl[cond]  # (n',)

        unravel = unravel[:, *((None,)*len(poni_b)), :].expand(-1, *poni_b, -1)  # (n, ..., o)
        unravel = unravel[cond]  # (n', o)
        if poni_b:
            indices = torch.meshgrid(  # (...,)
                *(torch.arange(b, device=unravel.device) for b in poni_b), indexing="ij"
            )
            indices = [  # (n, ...)
                idx.unsqueeze(0).expand(len(u_f), *((-1,)*len(poni_b))) for idx in indices
            ]
            unravel_poni = torch.cat([idx[cond].unsqueeze(1) for idx in indices], dim=1)  # (n', p)
            unravel = torch.cat([unravel, unravel_poni], dim=1)  # (n', o+p)

        return point, energy, hkl, unravel

    def compute_primitive_bc(self, **kwargs) -> torch.Tensor:
        r"""Get \(\mathbf{A}\) of shape (..., 3, 3) in the crystal base \(\mathcal{B^c}\)."""
        if (lattice := self._select_lattice(kwargs)) is None:
            raise AttributeError("`lattice` parameter has to be supplied")
        if lattice.requires_grad:
            return lattice_to_primitive(lattice)
        signature = ("primitive_bc", lattice.data_ptr())
        if (primitive_bc := self._cache.get(signature, None)) is None:
            primitive_bc = self._cache[signature] = lattice_to_primitive(lattice)
        return primitive_bc

    def compute_reciprocal_bc(self, **kwargs) -> torch.Tensor:
        r"""Get \(\mathbf{B}\) of shape (..., 3, 3) in the crystal base \(\mathcal{B^c}\)."""
        primitive_bc = self.compute_primitive_bc(**kwargs)
        if primitive_bc.requires_grad:
            return primitive_to_reciprocal(primitive_bc)
        signature = ("reciprocal_bc", primitive_bc.data_ptr())
        if (reciprocal_bc := self._cache.get(signature, None)) is None:
            reciprocal_bc = self._cache[signature] = primitive_to_reciprocal(primitive_bc)
        return reciprocal_bc

    def compute_reciprocal_bl(self, **kwargs) -> torch.Tensor:
        r"""Get \(\mathbf{B}\) of shape (..., 3, 3) in the lab base \(\mathcal{B^l}\)."""
        reciprocal_bc = self.compute_reciprocal_bc(**kwargs)
        rot = self.compute_rot(**kwargs)
        if rot.requires_grad or reciprocal_bc.requires_grad:
            return rotate_crystal(reciprocal_bc, rot, cartesian_product=False)
        signature = ("reciprocal_bl", reciprocal_bc.data_ptr(), rot.data_ptr())
        if (reciprocal_bl := self._cache.get(signature, None)) is None:
            reciprocal_bl = self._cache[signature] = (
                rotate_crystal(reciprocal_bc, rot, cartesian_product=False)
            )
        return reciprocal_bl

    def compute_rot(self, **kwargs) -> torch.Tensor:
        """Get the rotation matrix of shape (..., 3, 3)."""
        if (omega := self._select_omega(kwargs)) is None:
            raise AttributeError("`omega` parameter has to be supplied")
        if omega.requires_grad:
            return omega_to_rot(*omega.movedim(-1, 0), cartesian_product=False)
        signature = ("rot", omega.data_ptr())
        if (rot := self._cache.get(signature, None)) is None:
            rot = self._cache[signature] = (
                omega_to_rot(*omega.movedim(-1, 0), cartesian_product=False)
            )
        return rot

    def compute_uf(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Get the \(u_f\) unitary vector in the lab base \(\mathcal{B^l}\)."""
        u_q, energy, hkl, unravel = self.compute_uq(**kwargs)
        if u_q.requires_grad:
            return uq_to_uf(u_q), energy, hkl, unravel
        signature = ("uf", u_q.data_ptr())
        if (u_f := self._cache.get(signature, None)) is None:
            u_f = self._cache[signature] = uq_to_uf(u_q)
        return u_f, energy, hkl, unravel

    def compute_uq(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Get the \(u_q\) unitary vector in the lab base \(\mathcal{B^l}\).

        Returns
        -------
        u_q : torch.Tensor
            The ravel \(u_q\) of shape (n, 3).
        energy : torch.Tensor
            The energy in J of each \(u_q\) of shape (n,).
        hkl : torch.Tensor
            The full hkl list of shape (n, 3).
        unravel : torch.Tensor
            The int64 indice of each batch dimension of shape (n, o).
            It means that the ith elements comme from the batch unravel[i].
            The dimensions are the batch dimension of ``self.reciprocal_bl``.
        """
        hkl = self._compute_candidate_hkl(**kwargs)  # (n, 3)
        reciprocal_bl = self.compute_reciprocal_bl(**kwargs)  # (..., 3, 3)
        e_min = self._select_hkl_args(kwargs).get("e_min", 0.0)

        def compute_uq_(hkl, reciprocal_bl, e_min):
            u_q, energy = hkl_reciprocal_to_uq_energy(hkl, reciprocal_bl, cartesian_product=True)
            cond = energy >= e_min  # (m, ...)
            u_q, energy = u_q[cond], energy[cond]
            indices = torch.meshgrid(
                *(torch.arange(b, device=hkl.device) for b in cond.shape), indexing="ij"
            )
            unravel = torch.cat([idx[cond].unsqueeze(1) for idx in indices], dim=1)
            hkl, unravel = hkl[unravel[:, 0]], unravel[:, 1:]
            return u_q, energy, hkl, unravel

        if reciprocal_bl.requires_grad:
            return compute_uq_(hkl, reciprocal_bl, e_min)
        signature = ("uq", hkl.data_ptr(), reciprocal_bl.data_ptr(), e_min)
        if (res := self._cache.get(signature, None)) is None:
            res = self._cache[signature] = compute_uq_(hkl, reciprocal_bl, e_min)
        return res

    @property
    def cam_size(self) -> tuple[float, float]:
        """Return the camera size."""
        return self._cam_size

    @property
    def e_min(self) -> float:
        """Return the low energy limit in J."""
        return self._hkl_args.get("e_min", 0.0)

    @e_min.setter
    def e_min(self, e_min: None | numbers.Real):
        """Set the low energy limit."""
        if e_min is None:
            if "e_min" in self._hkl_args:
                del self._hkl_args["e_min"]
        else:
            assert isinstance(e_min, numbers.Real), e_min.__class__.__name__
            assert e_min >= 0.0, e_min
            self._hkl_args["e_min"] = float(e_min)

    @property
    def e_max(self) -> float:
        """Return the high energy limit in J."""
        return self._hkl_args.get("e_max", torch.inf)

    @e_max.setter
    def e_max(self, e_max: None | numbers.Real):
        """Set the high energy limit."""
        if e_max is None:
            if "e_max" in self._hkl_args:
                del self._hkl_args["e_max"]
        else:
            assert isinstance(e_max, numbers.Real), e_max.__class__.__name__
            assert e_max >= self.e_min, e_max
            self._hkl_args["e_max"] = float(e_max)

    def forward(self):
        """Fake method to solve abstract class initialisation."""
        raise NotImplementedError("the forward method has to be overwritten")

    @property
    def lattice(self) -> None | torch.Tensor:
        """Return the lattice parameters of shape (..., 6)."""
        return self._lattice

    @lattice.setter
    def lattice(self, lattice: None | torch.Tensor):
        """Set the lattice parameters."""
        if lattice is None:
            self._lattice = None
        else:
            assert isinstance(lattice, torch.Tensor), lattice.__class__.__name__
            assert lattice.shape[-1:] == (6,), lattice.shape
            self._lattice = torch.nn.Parameter(lattice)

    @property
    def omega(self) -> None | torch.Tensor:
        """Get the rotation angles of the different grains of shape (..., 3)."""
        return self._omega

    @omega.setter
    def omega(self, omega: None | torch.Tensor):
        """Set the rotation angles."""
        if omega is None:
            self._omega = None
        else:
            assert isinstance(omega, torch.Tensor), omega.__class__.__name__
            assert omega.shape[-1:] == (3,), omega.shape
            self._omega = torch.nn.Parameter(omega)

    @property
    def poni(self) -> None | torch.Tensor:
        """Return the camera calibration parameters of shape (..., 6)."""
        return self._poni

    @poni.setter
    def poni(self, poni: None | torch.Tensor):
        """Set the camera calibration parameters."""
        if poni is None:
            self._poni = None
        else:
            assert isinstance(poni, torch.Tensor), poni.__class__.__name__
            assert poni.shape[-1:] == (6,), poni.shape
            self._poni = torch.nn.Parameter(poni)

    @property
    def primitive_bc(self) -> None | torch.Tensor:
        """Alias to ``laueimproc.geometry.link.Geometry.compute_primitive_bc``."""
        try:
            return self.compute_primitive_bc()
        except AttributeError:
            return None

    @property
    def reciprocal_bc(self) -> None | torch.Tensor:
        """Alias to ``laueimproc.geometry.link.Geometry.compute_reciprocal_bc``."""
        try:
            return self.compute_reciprocal_bc()
        except AttributeError:
            return None

    @property
    def reciprocal_bl(self) -> None | torch.Tensor:
        """Alias to ``laueimproc.geometry.link.Geometry.compute_reciprocal_bl``."""
        try:
            return self.compute_reciprocal_bl()
        except AttributeError:
            return None

    @property
    def rot(self) -> None | torch.Tensor:
        """Alias to ``laueimproc.geometry.link.Geometry.compute_rot``."""
        try:
            return self.compute_rot()
        except AttributeError:
            return None

    @property
    def u_f(self) -> None | torch.Tensor:
        """Alias to ``laueimproc.geometry.link.Geometry.compute_uf``."""
        try:
            return self.compute_uf()[0]
        except AttributeError:
            return None

    @property
    def u_q(self) -> None | torch.Tensor:
        """Alias to ``laueimproc.geometry.link.Geometry.compute_uq``."""
        try:
            return self.compute_uq()[0]
        except AttributeError:
            return None
