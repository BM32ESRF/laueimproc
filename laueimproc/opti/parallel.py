#!/usr/bin/env python3

"""Manage the auto multithreading."""


import functools
import hashlib
import multiprocessing.pool
import os
import pickle
import threading
import time

from laueimproc.opti.cache import auto_cache
from laueimproc.opti.manager import DiagramManager
from laueimproc.opti.singleton import MetaSingleton


class ThreadManager(threading.Thread, metaclass=MetaSingleton):
    """Excecute several functions in parallel threads."""

    def __init__(self, verbose=False):
        self.jobs = {}  # each signature, associate the thread
        self.lock = threading.Lock()
        super().__init__(daemon=True)  # daemon has to be true to allow to exit python
        self.start()

    def contains(self, signature):
        with self.lock:
            return signature in self.jobs

    def run(self):
        """Asynchrone control loop."""
        while True:
            # starts waiting threads
            with self.lock:
                running = [job for job in self.jobs.values() if job.is_alive()]
                if nbr_to_append := max(0, os.cpu_count()-len(running)):
                    for job in self.jobs.values():
                        if not job._started.is_set():
                            job.start()
                            if (nbr_to_append := nbr_to_append-1) == 0:
                                break

            # transfere finish thread in cache
            with self.lock:
                finish = {
                    signature: job for signature, job in self.jobs.items()
                    if job._started.is_set() and not job.is_alive()
                }.items()
                for signature, job in finish:
                    with job.diagram._cache_lock:
                        job.diagram._cache[signature] = job.get()
                    del self.jobs[signature]

            # wait to free up resources
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
        return job.get()  # job is deleted from `run`.


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
        # case cached
        param_sig = hashlib.md5(pickle.dumps((args, kwargs)), usedforsecurity=False).hexdigest()
        signature = f"thread_{meth.__name__}_{diagram.signature}_{param_sig}"
        with diagram._cache_lock:  # pylint: disable=W0212
            if signature in diagram._cache:  # pylint: disable=W0212
                return diagram._cache.pop(signature)  # pylint: disable=W0212
        # case thread calculus
        job = Calculator(meth, diagram, args, kwargs)
        job.start()
        thread_manager = ThreadManager()
        for other_diagram in DiagramManager().get_nexts_diagrams(diagram, nbr=os.cpu_count()):
            thread_manager.submit_job(
                meth, other_diagram, args, kwargs,
                signature=f"thread_{meth.__name__}_{other_diagram.signature}_{param_sig}",
            )
        return job.get()

    return multithreaded_meth
