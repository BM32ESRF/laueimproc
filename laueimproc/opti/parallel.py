#!/usr/bin/env python3

"""Manage the auto multithreading."""


import functools
import hashlib
import multiprocessing.pool
import os
import pickle
import queue
import threading
import time

from laueimproc.opti.manager import DiagramManager
from laueimproc.opti.singleton import MetaSingleton


class ThreadManager(threading.Thread, metaclass=MetaSingleton):
    """Excecute several functions in parallel threads."""

    def __init__(self, verbose=False):
        self.jobs = {}  # each signature, associate the tread
        self.lock = threading.Lock()
        super().__init__(daemon=True)  # daemon has to be true to allow to exit python
        self.start()

    def contains(self, signature):
        with self.lock:
            return signature in self.jobs

    def run(self):
        """Asynchrone control loop."""
        while True:
            with self.lock:
                running = [job for job in self.jobs.values() if job.is_alive()]
                if nbr_to_append := max(0, os.cpu_count()-len(running)):
                    for job in self.jobs.values():
                        if not job._started.is_set():
                            job.start()
                            if (nbr_to_append := nbr_to_append-1) == 0:
                                break
            job = running.pop(0) if running else None
            if job is not None:
                job.join()  # passive waiting
            else:
                time.sleep(0.01)

    def submit_job(self, meth, diagram, args, kwargs, signature):
        with self.lock:
            if signature not in self.jobs:
                self.jobs[signature] = Calculator(meth, diagram, args, kwargs)
            return self.jobs[signature]

    def get_job(self, meth, diagram, args, kwargs, signature):
        job = self.submit_job(meth, diagram, args, kwargs, signature)
        with self.lock:
            if not job._started.is_set():
                job.start()
        result = job.get()
        with self.lock:
            del self.jobs[signature]
        return result


class Calculator(threading.Thread):
    def __init__(self, meth, diagram, args, kwargs):
        self.meth, self.diagram, self.args, self.kwargs = meth, diagram, args, kwargs
        self.result = self.exception = None
        super().__init__(daemon=True)

    def run(self):
        try:
            self.result = self.meth(self.diagram, *self.args, **self.kwargs)
        except Exception as err:
            self.exception = err

    def get(self):
        self.join()
        if self.exception is not None:
            raise self.exception
        return self.result


def auto_parallel(meth: callable) -> callable:
    """Decorator to auto multithread a Diagram method."""
    assert hasattr(meth, "__call__"), meth.__class__.__name__
    @functools.wraps(meth)
    def multithreaded_meth(diagram, *args, parallel: bool = True, **kwargs):
        # case no threaded calculus
        assert isinstance(parallel, bool), parallel.__class__.__name__
        if not parallel or threading.current_thread().name != "MainThread":
            return meth(diagram, *args, **kwargs)
        # case thread calculus
        param_sig = hashlib.md5(pickle.dumps((args, kwargs))).hexdigest()
        signature = f"{meth.__name__}_{id(diagram)}_{param_sig}"
        thread_manager = ThreadManager()
        if not thread_manager.contains(signature):
            for other_diagram in DiagramManager().diagrams:
                thread_manager.submit_job(
                    meth, other_diagram, args, kwargs,
                    signature=f"{meth.__name__}_{id(other_diagram)}_{param_sig}",
                )
        return thread_manager.get_job(meth, diagram, args, kwargs, signature)

    return multithreaded_meth
