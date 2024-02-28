#!/usr/bin/env python3

"""Group of diagrams."""


import gc
import sys
import threading
import time

from laueimproc.opti.singleton import MetaSingleton



class DiagramManager(threading.Thread, metaclass=MetaSingleton):
    """Manage a group of diagram asynchronousely.'

    Parameters
    ----------
    verbose : boolean
        The chatting status of the experiment (read and write).
    """

    def __init__(self, verbose=False):
        # declaration
        self._diagrams: dict = {}  # all the diagrams
        self._diagrams_lock: threading.Lock = threading.Lock()
        self._verbose: bool

        # initialisation
        assert isinstance(verbose, bool), verbose.__class__.__name__
        self._verbose = verbose

        # start thread
        super().__init__(daemon=True)  # daemon has to be true to allow to exit python
        self.start()

    def run(self):
        """Asynchrone control loop."""
        while 1:
            self.remove_unreferenced_diagrams()
            time.sleep(1)

    def add_diagram(self, diagram):
        """Add a new diagram into the set of diagrams.

        Paremeters
        ----------
        new_diagram : laueimproc.classes.base_diagram.BaseDiagram
        """
        new_stats = {"last_action": time.time()}
        with self._diagrams_lock:
            self._diagrams[diagram] = new_stats
            if self._verbose:
                print(f"diagram {id(diagram)} added to manager (there are {len(self._diagrams)})")

    def remove_unreferenced_diagrams(self):
        """Remove all the dereferenced diagrams."""
        to_delete = []
        with self._diagrams_lock:
            for diagram in self._diagrams:
                if (
                    sys.getrefcount(diagram) - len(diagram._spots or [])  # pylint: disable=W0212
                ) <= 3:  # ref to self._diagrams, sys.getrefcount, local var diagram
                    to_delete.append(diagram)
            for diagram in to_delete:
                del self._diagrams[diagram]
                if self._verbose:
                    print(
                        f"diagram {id(diagram)} removed from manager "
                        f"({len(self._diagrams)} left)"
                    )

    @property
    def verbose(self) -> bool:
        """Get the chatting status of the experiment."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """Set the chatting status of the experiment."""
        assert isinstance(verbose, bool), verbose.__class__.__name__
        self._verbose = verbose
