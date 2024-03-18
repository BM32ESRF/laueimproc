#!/usr/bin/env python3

"""Help to manage the cache."""

import functools
import hashlib
import pickle
import sys

import torch


def auto_cache(meth: callable) -> callable:
    """Decorator to manage the cache of a Diagram method."""
    assert hasattr(meth, "__call__"), meth.__class__.__name__
    @functools.wraps(meth)
    def cached_meth(diagram, *args, is_cached: bool = False, **kwargs):
        param_sig = hashlib.md5(pickle.dumps((args, kwargs)), usedforsecurity=False).hexdigest()
        signature = f"cache: {diagram.state}.{meth.__name__}({param_sig})"
        if is_cached:  # no lock because operation is atomic
            return signature in diagram._cache  # pylint: disable=W0212
        with diagram._cache_lock:  # pylint: disable=W0212
            if signature in diagram._cache:  # pylint: disable=W0212
                return diagram._cache[signature]  # pylint: disable=W0212
        res = meth(diagram, *args, **kwargs)
        with diagram._cache_lock:  # pylint: disable=W0212
            diagram._cache[signature] = res  # pylint: disable=W0212
        return res
    return cached_meth


def getsizeof(obj: object, *, _processed: set = None) -> int:
    """Recursive version of sys.getsizeof."""
    _processed = _processed or set()
    if id(obj) in _processed:
        return 0
    mem = sys.getsizeof(obj)
    _processed.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return mem + obj.element_size() * obj.nelement()
    if isinstance(obj, (list, tuple, set, frozenset)):
        return mem + sum(getsizeof(o, _processed=_processed) for o in obj)
    if isinstance(obj, dict):
        return mem + sum(
            getsizeof(k, _processed=_processed) + getsizeof(v, _processed=_processed)
            for k, v in obj.items()
        )
    return mem
