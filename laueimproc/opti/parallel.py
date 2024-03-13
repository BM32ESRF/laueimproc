#!/usr/bin/env python3

"""Manage the auto multithreading."""


import functools
import hashlib
import logging
import os
import pickle
import queue
import threading

from laueimproc.opti.manager import DiagramManager
from laueimproc.opti.singleton import MetaSingleton


class ThreadManager(threading.Thread, metaclass=MetaSingleton):
    """Excecute several functions in parallel threads."""

    def __init__(self, verbose=False):
        self.jobs = {}  # each signature, associate the thread
        self.lock = threading.Lock()
        self.submit_event = queue.Queue(maxsize=1)  # faor passive waiting base on event
        super().__init__(daemon=True)  # daemon has to be true to allow to exit python
        self.start()

    def run(self):
        """Asynchrone control loop."""
        while True:
            # starts waiting threads
            with self.lock:
                running = [job for job in self.jobs.values() if job.is_alive()]
                if nbr_to_append := max(0, 3*os.cpu_count()//2-len(running)):
                    for job in self.jobs.values():
                        if not job._started.is_set():
                            job.start()
                            if (nbr_to_append := nbr_to_append-1) == 0:
                                break

            # transfere terminated result thread in cache
            with self.lock:
                finish = {
                    signature: job for signature, job in self.jobs.items()
                    if job._started.is_set() and not job.is_alive()
                }
                for signature in finish:
                    del self.jobs[signature]
            for signature, job in finish.items():
                with job.diagram._cache_lock:
                    try:
                        job.diagram._cache[signature] = job.get()
                    except Exception as err:  # assumed to be durty
                        logging.warning(err)  # dont raise because could come from not thread safe

            # wait to free up resources
            job = running.pop(0) if running else None
            if job is not None:
                job.join()  # passive waiting
            else:
                try:
                    self.submit_event.get(timeout=0.5)
                except queue.Empty:  # beter than a short time.sleep active waiting
                    pass

    def submit_job(self, meth, diagram, args, kwargs, signature):
        if not self.is_alive():
            raise RuntimeError("the thread manager is dead")
        with self.lock:
            if signature not in self.jobs:
                self.jobs[signature] = Calculator(meth, diagram, args, kwargs)
            job = self.jobs[signature]
        try:
            self.submit_event.put_nowait(None)  # just for trigger main thread
        except queue.Full:
            pass
        return job

    def pop(self, meth, diagram, args, kwargs, signature):
        job = self.submit_job(meth, diagram, args, kwargs, signature)
        with self.lock:
            if not job._started.is_set():
                job.start()
        # return job.get()
        res = job.get()
        with self.lock:  # delete all references
            if signature in self.jobs:
                del self.jobs[signature]
        with diagram._cache_lock:
            if signature in diagram._cache:
                del diagram._cache[signature]
        return res


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
        signature = f"thread: {diagram.signature}.{meth.__name__}({param_sig})"
        with diagram._cache_lock:  # pylint: disable=W0212
            if signature in diagram._cache:  # pylint: disable=W0212
                return diagram._cache.pop(signature)  # pylint: disable=W0212
        # case thread calculus
        thread_manager = ThreadManager()
        for next_diagram in DiagramManager().get_nexts_diagrams(diagram, 3*os.cpu_count()):
            next_signature = f"thread: {next_diagram.signature}.{meth.__name__}({param_sig})"
            with next_diagram._cache_lock:  # pylint: disable=W0212
                if next_signature not in next_diagram._cache:  # pylint: disable=W0212
                    thread_manager.submit_job(meth, next_diagram, args, kwargs, next_signature)
        return thread_manager.pop(meth, diagram, args, kwargs, signature)

    return multithreaded_meth
