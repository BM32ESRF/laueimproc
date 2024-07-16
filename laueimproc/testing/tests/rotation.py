#!/usr/bin/env python3

"""Test the rotation function."""

import torch
from laueimproc.geometry.rotation import omega_to_rot, rot_to_omega


def test_bij_omega_to_rot_to_omega():
    """Test if the function is bijective."""
    theta1 = 2.0 * torch.pi * torch.rand(1_000_000, dtype=torch.float64) - torch.pi
    theta2 = torch.pi * torch.rand(1_000_000, dtype=torch.float64) - torch.pi / 2
    theta3 = 2.0 * torch.pi * torch.rand(1_000_000, dtype=torch.float64) - torch.pi
    rot = omega_to_rot(theta1, theta2, theta3, cartesian_product=False)
    theta_bis = rot_to_omega(rot)
    assert torch.allclose(theta1, theta_bis[:, 0])
    assert torch.allclose(theta2, theta_bis[:, 1])
    assert torch.allclose(theta3, theta_bis[:, 2])
