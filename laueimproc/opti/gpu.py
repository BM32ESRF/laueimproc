#!/usr/bin/env python3

"""Manage the auto gpu."""

import torch


def to_device(obj: object, device: str) -> object:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (tuple, list, set, frozenset)):
        return type(obj)(to_device(e, device) for e in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj

def auto_gpu(func):
    def func_gpu(*args, **kwargs):
        args = to_device(args, "cuda")
        kwargs = to_device(kwargs, "cuda")
        res = func(*args, **kwargs)
        args = to_device(args, "cpu")
        kwargs = to_device(kwargs, "cpu")
        return res
    return func_gpu
