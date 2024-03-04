#!/usr/bin/env python3

"""Group of diagrams."""

import collections
import gc
import numbers
import threading
import time

from laueimproc.opti.singleton import MetaSingleton
from laueimproc.opti.memory import get_swappiness, mem_to_free


class DiagramManager(threading.Thread, metaclass=MetaSingleton):
    """Manage a group of diagram asynchronousely.

    Parameters
    ----------
    verbose : boolean
        The chatting status of the experiment (read and write).
    ram_limit : float
        The maximum amount in percent of memory before trying to release some cache.
    """

    def __init__(self, verbose=False):
        # declaration
        self._diagrams: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()
        self._ram_limit: float = float((100-get_swappiness())) / 100.0
        self._verbose: bool

        # initialisation
        assert isinstance(verbose, bool), verbose.__class__.__name__
        self._verbose = verbose

        # start thread
        super().__init__(daemon=True)  # daemon has to be true to allow to exit python
        self.start()

    def add_diagram(self, diagram):
        """Track the new diagram."""
        with self._lock:
            self._diagrams[diagram] = None

    @property
    def diagrams(self) -> list:
        """Return the diagrams in a list."""
        with self._lock:
            return list(self._diagrams)  # slow but thread safe

    @property
    def ram_limit(self) -> float:
        """Return the threashold of ram in percent."""
        return self._ram_limit

    @ram_limit.setter
    def ram_limit(self, new_limit: float):
        assert isinstance(new_limit, numbers.Real)
        assert 0 < new_limit < 1, new_limit
        self._ram_limit = float(new_limit)

    def run(self):
        """Asynchrone control loop."""
        while True:
            # memory free
            with self._lock:
                if self._diagrams and (size := mem_to_free(self._ram_limit) // len(self._diagrams)):
                    if self._verbose:
                        print(f"try to clear {size} bytes of cache")
                    for diagram in self._diagrams:
                        diagram.clear_cache(size)
            time.sleep(1)

    def update(self):
        """Try to keep only reachable diagrams."""
        from laueimproc.classes.diagram import Diagram  # pylint: disable=C0415
        with self._lock:
            before = len(self._diagrams)
            order = [id(diagram) for diagram in self._diagrams]
            self._diagrams = collections.OrderedDict()
            gc.collect()
            new_diagrams = {id(o): o for o in gc.get_objects() if isinstance(o, Diagram)}
            self._diagrams = collections.OrderedDict(
                (new_diagrams[id_], None) for id_ in order if id_ in new_diagrams
            )
            after = len(self._diagrams)
        if self._verbose:
            print(f"update tracked diagrams, {before-after} are freed, {after} remain")

    @property
    def verbose(self) -> bool:
        """Get the chatting status of the experiment."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """Set the chatting status of the experiment."""
        assert isinstance(verbose, bool), verbose.__class__.__name__
        self._verbose = verbose
