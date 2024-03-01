#!/usr/bin/env python3

"""Group of diagrams."""

import threading
import time

from laueimproc.opti.singleton import MetaSingleton
from laueimproc.opti.cache import getsizeof
from laueimproc.opti.memory import mem_to_free


class DiagramManager(threading.Thread, metaclass=MetaSingleton):
    """Manage a group of diagram asynchronousely.

    Parameters
    ----------
    verbose : boolean
        The chatting status of the experiment (read and write).
    """

    def __init__(self, verbose=False):
        # declaration
        self._verbose: bool
        self._diagrams: list = []  # memory id -> diagram
        self._lock = threading.Lock()

        # initialisation
        assert isinstance(verbose, bool), verbose.__class__.__name__
        self._verbose = verbose

        # start thread
        super().__init__(daemon=True)  # daemon has to be true to allow to exit python
        self.start()

    def add_diagram(self, diagram):
        with self._lock:
            self._diagrams.append(diagram)

    def run(self):
        """Asynchrone control loop."""
        while True:
            # memory free
            with self._lock:
                if self._diagrams and (size := mem_to_free() // len(self._diagrams)):
                    for diagram in self._diagrams:
                        diagram.clear_cache(size)
            time.sleep(1)

    @property
    def diagrams(self) -> list:
        """Return the diagrams in a list."""
        with self._lock:
            return self._diagrams.copy()  # slow but thread safe
        # from laueimproc.classes.diagram import Diagram
        # return [o for o in gc.get_objects() if isinstance(o, Diagram)]

    @property
    def verbose(self) -> bool:
        """Get the chatting status of the experiment."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """Set the chatting status of the experiment."""
        assert isinstance(verbose, bool), verbose.__class__.__name__
        self._verbose = verbose
