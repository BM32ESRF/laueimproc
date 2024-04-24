#!/usr/bin/env python3

"""Test if the dependents libraries are well installed and linked.

Basicaly, it checks if the installation seems to be correct.
"""

import subprocess

import torch


def test_gpu_torch():
    """Test if torch is able to use the GPU."""
    # possible to test lspci | grep ' NVIDIA '
    if torch.cuda.is_available():
        return  # case always ok
    try:
        result = subprocess.run(["lshw", "-C", "display"], capture_output=True, check=True)
    except FileNotFoundError as err:
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        except FileNotFoundError:
            return  # assume there are not graphical card
        raise ImportError(
            "There seems to be an nvidia gpu on this machine, "
            "however torch is not able to use it, please reinstall cuda"
        ) from err
    if b" nvidia " in result.stdout.lower():
        raise ImportError(
            "There seems to be an nvidia gpu on this machine, "
            "however torch is not able to use it, please reinstall cuda"
        )
