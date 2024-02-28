#!/usr/bin/env python3

"""Allow to create only one instance of an object."""


class MetaSingleton(type):
    """For share memory inside the current session.

    If a class inerits from this metaclass, only one instance of the class can exists.
    If we try to create a new instnce, it will return a reference to the unique instance.
    """

    instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        """Create a new class only if it is not already instanciated."""
        if cls not in MetaSingleton.instances:
            instance = cls.__new__(cls)
            instance.__init__(*args, **kwargs)
            MetaSingleton.instances[cls] = instance
        return MetaSingleton.instances[cls]
