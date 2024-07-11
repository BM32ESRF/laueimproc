#!/usr/bin/env python3

"""Full model for complete simulation.

Combines the various elementary functions into a torch module.
"""

import abc

import torch

from .bragg import hkl_reciprocal_to_uq_energy, uq_to_uf
from .hkl import select_hkl
from .lattice import lattice_to_primitive
from .projection import ray_to_detector
from .reciprocal import primitive_to_reciprocal
from .rotation import rotate_crystal, rot_to_angle


class GeneralSimulator(torch.nn.Module, abc.ABC):
    """General simulation class.

    Attributes
    ----------
    lattice : None | torch.Tensor
        The lattice parameters of shape (..., 6).
    phi : None | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The decomposition of rot in elementary angles.
    rot : None | torch.Tensor
        The rotation matrix of shape (..., 3, 3).
    """

    _lattice = None
    _rot = None

    @property
    def lattice(self) -> None | torch.Tensor:
        """Return the lattice parameters of shape (..., 6)."""
        return self._lattice

    @property
    def phi(self) -> None | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the decomposition of rot in elementary angles."""
        if (rot := self.rot) is None:
            return None
        return rot_to_angle(rot)

    @property
    def rot(self) -> None | torch.Tensor:
        """Return the rotation matrix of shape (..., 3, 3)."""
        return self._rot


class SimulatorUf2Cam(GeneralSimulator):
    """Simulation from u_f to camera."""

    def __init__(self, lattice: torch.Tensor, rot: torch.Tensor):
        super().__init__()
        assert isinstance(lattice, torch.Tensor)
        assert lattice.shape[-1:] == (6,), lattice.shape
        assert isinstance(rot, torch.Tensor)
        assert rot.shape[-2:] == (3, 3), rot.shape

        self._lattice = lattice
        self._rot = rot

    @property
    def u_f(self) -> torch.Tensor:
        r"""Return the \(u_f\) vector."""
        raise NotImplementedError

    def forward(self):
        """Simulate."""
        raise NotImplementedError


# class Simulator:
#     """Simulate a full multigrain laue diagram."""

#     def __new__(
#         cls,
#         lattice: torch.Tensor | None = None,
#         rot: torch.Tensor | None = None,
#         poni: torch.Tensor | None = None,
#     ):
#         """Select the model.

#         Parameters
#         ----------
#         lattice : torch.Tensor, optional
#             The lattice parameters of the grains to be simulated, shape (..., 6).
#         rot : torch.Tensor, optional
#             The rotation matrix of the different grains, shape (..., 3, 3).
#         poni : torch.Tensor, optional
#             The camera calibration parameters, shape (..., 6).
#         """
#         match (lattice is None, rot is None, poni is None):
#             case (False, False, True):
#                 return SimulatorUf2Cam(lattice, rot)
#             case _:
#                 raise NotImplementedError("this configuration is not yet implemented")


# class FullSimulator(torch.nn.Module):
#     """Full simulation from lattice parameters to final laue diagram."""

#     def __init__(
#         self,
#         lattice: torch.Tensor | None = None,
#         rot: torch.Tensor | None = None,
#         poni: torch.Tensor | None = None,
#     ):
#         """Initialise the model.

#         Parameters
#         ----------
#         lattice : torch.Tensor, optional
#             The lattice parameters of the grains to be simulated, shape (..., 6).
#         rot : torch.Tensor, optional
#             The rotation matrix of the different grains, shape (..., 3, 3).
#         poni : torch.Tensor, optional
#             The camera calibration parameters, shape (..., 6).
#         """
#         super().__init__()
#         if lattice is not None:
#             assert isinstance(lattice, torch.Tensor), lattice.__class__.__name__
#             assert lattice.shape[-1:] == (6,), lattice.shape
#         if rot is not None:
#             assert isinstance(rot, torch.Tensor), rot.__class__.__name__
#             assert rot.shape[-2:] == (3, 3), rot.shape
#         if poni is not None:
#             assert isinstance(poni, torch.Tensor), poni.__class__.__name__
#             assert poni.shape[-1:] == (6,), poni.shape

#         self._lattice = lattice
#         self._rot = rot
#         self._poni = poni

#     def forward(self) -> torch.Tensor:
#         """Simulate the grains.

#         Examples
#         --------
#         >>> import torch
#         >>> from laueimproc.geometry.model import FullSimulator
#         >>> lattice = torch.tensor([3.6e-10, 3.6e-10, 3.6e-10, torch.pi/2, torch.pi/2, torch.pi/2])
#         >>> poni = torch.tensor([0.07, 73.4e-3, 73.4e-3, 0.0, -torch.pi/2, 0.0])
#         >>> rot = torch.eye(3)
#         >>> simulator = FullSimulator(lattice, rot, poni)
#         >>> # simulator()
#         """
#         primitive = lattice_to_primitive(self._lattice)
#         reciprocal = primitive_to_reciprocal(primitive)
#         reciprocal = rotate_crystal(reciprocal, self._rot, cartesian_product=False)
#         hkl = select_hkl(reciprocal, e_max=20e3 * 1.60e-19)  # 20 keV
#         u_q, energy = hkl_reciprocal_to_uq_energy(hkl, reciprocal, cartesian_product=False)
#         u_f = uq_to_uf(u_q)
#         point, dist = ray_to_detector(u_f, self._poni, cartesian_product=False)

#         cond = dist > 0
#         hkl, energy, point = hkl[cond], energy[cond], point[cond]

#         return hkl, energy, point
