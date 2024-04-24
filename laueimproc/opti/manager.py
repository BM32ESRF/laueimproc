#!/usr/bin/env python3

"""Group of diagrams."""

import gc
import numbers
import threading
import time

from laueimproc.common import bytes2human
from laueimproc.opti.memory import free_malloc, get_swappiness, mem_to_free
from laueimproc.opti.singleton import MetaSingleton


class DiagramManager(threading.Thread, metaclass=MetaSingleton):
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
        self._diagrams_dict: dict = {}  # dict for fast acces
        self._diagrams_list: list = []  # list for order
        self._lock = threading.Lock()
        self._max_mem_percent: int = 100-get_swappiness()
        self._verbose: bool = False

        # start thread
        super().__init__(daemon=True)  # daemon has to be true to allow to exit python
        self.start()

    def add_diagram(self, diagram):
        """Track the new diagram."""
        with self._lock:
            if diagram not in self._diagrams_dict:
                self._diagrams_dict[diagram] = len(self._diagrams_list)
                self._diagrams_list.append(diagram)

    def collect(self) -> int:
        """Try to keep only reachable diagrams."""
        from laueimproc.classes.diagram import Diagram
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
                    total = 0
                    for level in (0, 1, 2):
                        total += sum(
                            d.compress((target-total)//nbr, _levels={level})
                            for d in self._diagrams_list
                        )
                        if total >= target:
                            break
                    if self._verbose:
                        print(f"{bytes2human(total)} of cache removed (level {level})")
                    free_malloc()
            time.sleep(0.1)

    @property
    def verbose(self) -> bool:
        """Get the chatting status of the experiment."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """Set the chatting status of the experiment."""
        assert isinstance(verbose, bool), verbose.__class__.__name__
        self._verbose = verbose


def collect():
    """Release all unreachable diagrams.

    Returns
    -------
    nbr : int
        The number of diagrams juste released.
    """
    return DiagramManager().collect()
