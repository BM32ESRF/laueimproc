#!/usr/bin/env python3

"""Refine omega and lattice parameters by maximizing the matching rate through gradient ascent."""

import numbers
import warnings

import torch

from ..bragg import hkl_reciprocal_to_uq_energy, uf_to_uq
from ..hkl import select_hkl
from ..lattice import lattice_to_primitive
from ..metric import compute_matching_score_continuous
from ..projection import detector_to_ray
from ..reciprocal import primitive_to_reciprocal
from ..rotation import omega_to_rot, rotate_crystal


class Refiner(torch.nn.Module):
    r"""Refine a single grain.

    Attributes
    ----------
    omega : torch.Tensor
        The 3 \(\omega\) elementary rotation angles of shape (3,).
    lattice : torch.Tensor
        The lattice parameters of shape (6,).
    """

    def __init__(
        self, lattice: torch.Tensor, omega: torch.Tensor, uf_or_point: torch.Tensor, **kwargs
    ):
        r"""Initialize the indexator.

        Parameters
        ----------
        lattice : torch.Tensor
            The relaxed lattice parameters of a single grain, of shape (6,).
        omega : torch.Tensor
            The 3 elementary rotation angles of shape (3,).
        uf_or_point : torch.Tensor
            The experimental data.
            The 2d point associated to uf in the referencial of the detector of shape (n, 2).
            Could be also the unitary vector \(u_q\) of shape (n, 3).
        e_min : float
            The minimum energy of the beam line in J.
        e_max : float
            The maximum energy of the beam line in J.
        poni: torch.Tensor, optional
            The 6 ordered .poni calibration parameters as a tensor of shape (6,).
            Used only if ``uf_or_point`` is a point.
        """
        assert set(kwargs).issubset({"e_min", "e_max", "poni"})
        assert isinstance(lattice, torch.Tensor), lattice.__class__.__name__
        assert lattice.shape == (6,), lattice.shape
        assert isinstance(omega, torch.Tensor), omega.__class__.__name__
        assert omega.shape == (3,), omega.shape
        assert isinstance(uf_or_point, torch.Tensor), uf_or_point.__class__.__name__
        assert uf_or_point.ndim == 2, uf_or_point.shape
        assert isinstance(kwargs["e_min"], numbers.Real), kwargs["e_min"].__class__.__name__
        assert 0.0 <= kwargs["e_min"], kwargs["e_min"]
        assert isinstance(kwargs["e_max"], numbers.Real), kwargs["e_max"].__class__.__name__
        assert kwargs["e_min"] < kwargs["e_max"] < torch.inf, (kwargs["e_min"], kwargs["e_max"])

        super().__init__()

        # convert point to uq
        poni = kwargs.get("poni", None)
        if uf_or_point.shape[-1] == 2:
            assert isinstance(poni, torch.Tensor), poni.__class__.__name__
            assert poni.shape == (6,), poni.shape
            uf_or_point = detector_to_ray(uf_or_point, poni)
        elif uf_or_point.shape[-1] == 3:
            if poni is not None:
                warnings.warn(
                    "the `poni` is ignored because uf has been provided, not point",
                    RuntimeWarning,
                )
        else:
            raise ValueError("only shape 2 and 3 allow")

        # set parameters
        self._uq_exp = torch.nn.Parameter(uf_to_uq(uf_or_point), requires_grad=False)
        self._lattice = torch.nn.Parameter(lattice.clone(), requires_grad=True)
        self._omega = torch.nn.Parameter(omega.clone(), requires_grad=True)
        self._hkl = torch.nn.Parameter(
            select_hkl(
                primitive_to_reciprocal(lattice_to_primitive(lattice)), e_max=kwargs["e_max"]
            ),
            requires_grad=False,
        )
        self._e_min = kwargs["e_min"]

    def forward(self, phi_max: numbers.Real) -> torch.Tensor:
        """Compute the continuous matching rate.

        Parameters
        ----------
        phi_max : float
            The maximum positive angular distance in radian
            to consider that the rays are closed enough.

        Returns
        -------
        rate : torch.Tensor
            The scalar continuous matching rate.
        """
        assert isinstance(phi_max, numbers.Real), phi_max.__class__.__name__
        assert phi_max > 0, phi_max

        reciprocal_bc = primitive_to_reciprocal(lattice_to_primitive(self._lattice))
        rot = omega_to_rot(*self._omega, cartesian_product=False)
        reciprocal_bl = rotate_crystal(reciprocal_bc, rot, cartesian_product=False)
        uq_theo, energy = hkl_reciprocal_to_uq_energy(  # (n, 3), (n,)
            self._hkl, reciprocal_bl, cartesian_product=False
        )
        uq_theo = uq_theo[energy >= self._e_min]  # (n', 3)
        matching_rate = compute_matching_score_continuous(self._uq_exp, uq_theo, phi_max)

        return matching_rate

    @property
    def lattice(self) -> torch.Tensor:
        """Return the lattice parameters of shape (6,)."""
        return self._lattice.detach()

    @property
    def omega(self) -> torch.Tensor:
        """Return the 3 elementary rotation angles of shape (3,)."""
        return self._omega.detach()

    def refine(
        self, phi_max: numbers.Real, refine_abc: bool = True, refine_shear: bool = True
    ) -> float:
        r"""Refine the \(\omega\) angles and the lattice parameters.

        Parameters
        ----------
        phi_max : float
            The maximum positive angular distance in radian
            to consider that the rays are closed enough.
        refine_abc : boolean, default=True
            If True, refine the 3 lattice parameters \([a, b, c]\).
            The initial and the final sum of these parameters is guaranteed to be the same.
        refine_shear : boolean, default=True
            If True, refine the 3 angle lattice parameters \([\alpha, \beta, \gamma]\).

        Returns
        -------
        rate : float
            The final matching rate after rafinement.

        Notes
        -----
        You can call this function several times to rafine more.

        Examples
        --------
        >>> import torch
        >>> from laueimproc.geometry.indexation.refine import Refiner
        >>> lattice = torch.tensor([3.6e-10, 3.6e-10, 3.6e-10, torch.pi/2, torch.pi/2, torch.pi/2])
        >>> e_min, e_max = 5e3 * 1.60e-19, 25e3 * 1.60e-19  # 5-25 keV
        >>> omega = torch.zeros(3)
        >>> uq_exp = torch.randn(1000, 3)
        >>> uq_exp /= torch.linalg.norm(uq_exp, dim=1, keepdims=True)
        >>> refiner = Refiner(lattice, omega, uq_exp, e_min=e_min, e_max=e_max)
        >>> rate = refiner.refine(0.5 * torch.pi/180)
        >>>
        """
        assert isinstance(phi_max, numbers.Real), phi_max.__class__.__name__
        assert phi_max > 0, phi_max
        assert isinstance(refine_abc, bool), refine_abc.__class__.__name__
        assert isinstance(refine_shear, bool), refine_shear.__class__.__name__

        self.to(torch.float64)  # for better gradient accuracy
        self._omega.requires_grad = True
        self._lattice.requires_grad = True
        abc_sum = float(torch.sum(self._lattice[:3]))

        for _ in range(100):
            # compute grad
            rate = self.forward(phi_max)
            self._omega.grad = self._lattice.grad = None
            rate.backward()
            self._omega.grad = self._omega.grad.nan_to_num(nan=0.0).clamp(-100.0, 100.0)
            self._lattice.grad = self._lattice.grad.nan_to_num(nan=0.0).clamp(-100.0, 100.0)
            # update parameters
            with torch.no_grad():
                self._omega += 8.72e-8 * self._omega.grad  # max 0.5 degrees after 100 iterations
                if refine_shear:
                    self._lattice[3:] += 8.72e-8 * self._lattice.grad[3:]
                if refine_abc:   # max 0.1% after 100 iters
                    self._lattice[:3] *= (1.0 + 1e-6 * self._lattice.grad[:3])
                    self._lattice[:3] += (abc_sum - torch.sum(self._lattice[:3])) / 3.0

        self._omega.grad = self._lattice.grad = None
        with torch.no_grad():
            return float(self.forward(phi_max))
