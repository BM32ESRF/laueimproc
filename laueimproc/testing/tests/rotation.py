#!/usr/bin/env python3

"""Test the rotation function."""

import torch
from laueimproc.geometry.rotation import angle_to_rot, rot_to_angle


def test_bij_angle_to_rot_to_angle():
    """Test if the function is bijective."""
    theta1 = 2.0 * torch.pi * torch.rand(1_000_000, dtype=torch.float64) - torch.pi
    theta2 = torch.pi * torch.rand(1_000_000, dtype=torch.float64) - torch.pi / 2
    theta3 = 2.0 * torch.pi * torch.rand(1_000_000, dtype=torch.float64) - torch.pi
    rot = angle_to_rot(theta1, theta2, theta3, meshgrid=False)
    theta1_bis, theta2_bis, theta3_bis = rot_to_angle(rot)
    assert torch.allclose(theta1, theta1_bis)
    assert torch.allclose(theta2, theta2_bis)
    assert torch.allclose(theta3, theta3_bis)
