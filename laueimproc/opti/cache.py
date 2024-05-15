#!/usr/bin/env python3

"""Help to manage the cache."""

import functools
import gc
import hashlib
import numbers
import pickle
import sys
import threading
import time
import typing

import torch

from laueimproc.common import bytes2human
from laueimproc.opti.memory import free_malloc, get_swappiness, mem_to_free
from laueimproc.opti.singleton import MetaSingleton


class CacheManager(threading.Thread, metaclass=MetaSingleton):
    """Manage a group of diagram asynchronousely.

    Parameters
    ----------
    verbose : boolean
        The chatting status of the experiment (read and write).
    max_mem_percent : int
        The maximum amount of memory percent before trying to release some cache (read and write).
        By default it is based on swapiness.
    """

    def __init__(self):
        # declaration
        self._diagrams_dict: dict = {}  # dict for fast acces diagram -> index
        self._diagrams_list: list = []  # list for order
        self._lock = threading.Lock()
        self._max_mem_percent: int = 100-get_swappiness()
        self._verbose: bool = False

        # start thread
        super().__init__(daemon=True)  # daemon has to be true to allow to exit python
        self.start()

    def _untrack(self, diagram):
        """Stop tracking this diagram."""
        with self._lock:
            if (index := self._diagrams_dict.get(diagram, None)) is not None:
                del self._diagrams_dict[diagram]
                self._diagrams_list[index] = None

    def collect(self) -> int:
        """Try to keep only reachable diagrams."""
        from laueimproc.classes.diagram import Diagram
        self.squeeze_untracked()
        with self._lock:
            before = len(self._diagrams_list)
            diagrams_id = [id(diagram) for diagram in self._diagrams_list]
            self._diagrams_dict = {}
            self._diagrams_list = []
            gc.collect()
            new_diagrams = {id(o): o for o in gc.get_objects() if isinstance(o, Diagram)}
            self._diagrams_list = [new_diagrams[id_] for id_ in diagrams_id if id_ in new_diagrams]
            self._diagrams_dict = {
                diagram: index for index, diagram in enumerate(self._diagrams_list)
            }
            after = len(self._diagrams_list)
        free_malloc()
        return before - after

    @property
    def max_mem_percent(self) -> float:
        """Return the threashold of ram in percent."""
        return self._max_mem_percent

    @max_mem_percent.setter
    def max_mem_percent(self, new_limit: numbers.Real):
        assert isinstance(new_limit, numbers.Real)
        assert 0 <= new_limit <= 100, new_limit
        self._max_mem_percent = round(new_limit)

    def run(self):
        """Asynchron control loop."""
        while True:
            # memory free
            with self._lock:
                nbr = len(self._diagrams_list)
                target = mem_to_free(self._max_mem_percent)
                if nbr and (target := mem_to_free(self._max_mem_percent)):
                    total = sum(  # remove obsolete cache
                        d.compress(target//nbr, _levels={0}) for d in self._diagrams_list
                    )
                    i = 0
                    while (total >= target and i < len(self._diagrams_list)):  # remove active cache
                        diagram = self._diagrams_list[i]
                        total += diagram.compress((target-total)//nbr, _levels={1})
                        if not diagram._cache[1]:  # pylint: disable=W0212
                            self._untrack(diagram)
                        i += 1
                    self.squeeze_untracked()
                    if self._verbose:
                        print(f"{bytes2human(total)} of cache removed")
                    free_malloc()

            time.sleep(0.1)

    def squeeze_untracked(self) -> int:
        """Update the indices to remove the None values."""
        new_index = 0
        diagrams_list = []
        with self._lock:
            before = len(self._diagrams_list)
            for diagram in self._diagrams_list:
                if diagram is None:
                    continue
                diagrams_list.append(diagram)
                self._diagrams_dict[diagram] = new_index
                new_index += 1
            self._diagrams_list = diagrams_list
            after = len(self._diagrams_list)
        return before - after

    def track(self, diagram):
        """Track the new diagram if it is not already tracked."""
        with self._lock:
            if diagram not in self._diagrams_dict:
                self._diagrams_dict[diagram] = len(self._diagrams_list)
                self._diagrams_list.append(diagram)

    @property
    def verbose(self) -> bool:
        """Get the chatting status of the experiment."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """Set the chatting status of the experiment."""
        assert isinstance(verbose, bool), verbose.__class__.__name__
        self._verbose = verbose


def auto_cache(meth: typing.Callable) -> typing.Callable:
    """Decorate to manage the cache of a Diagram method."""
    assert callable(meth), meth.__class__.__name__

    @functools.wraps(meth)
    def cached_meth(diagram, *args, cache: bool = True, **kwargs):
        assert isinstance(cache, bool), cache.__class__.__name__
        if not cache:
            return meth(diagram, *args, **kwargs)
        param_sig = hashlib.md5(pickle.dumps((args, kwargs)), usedforsecurity=False).hexdigest()
        signature = f"cache: {diagram.state}.{meth.__name__}({param_sig})"
        with diagram._cache[0]:  # pylint: disable=W0212
            if signature in diagram._cache:  # pylint: disable=W0212
                return diagram._cache[1][signature]  # pylint: disable=W0212
        res = meth(diagram, *args, **kwargs)
        with diagram._cache[0]:  # pylint: disable=W0212
            diagram._cache[1][signature] = res  # pylint: disable=W0212
        CacheManager().track(diagram)
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


def collect():
    """Release all unreachable diagrams.

    Returns
    -------
    nbr : int
        The number of diagrams juste released.
    """
    return CacheManager().collect()
