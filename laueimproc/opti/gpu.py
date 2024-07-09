#!/usr/bin/env python3

"""Manage the auto gpu."""

import torch


def to_device(obj: object, device: str) -> object:
    """Trensfer all tensors to the new device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list | tuple | set | frozenset):
        return type(obj)(to_device(e, device) for e in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj
