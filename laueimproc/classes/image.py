#!/usr/bin/env python3

"""Defines the basic structure of a spot, inerit from torch array."""

import numpy as np
import torch


class Image(torch.Tensor):
    """A General image.

    Attributes
    ----------
    matadata : object
        Any information to throw during the operations.
    """

    def __new__(cls, data, metadata: object = None):
        """Create an image.

        If the image is not in floating point, it is converting in float32
        and normalized beetweeen 0 and 1.

        Parameters
        ----------
        metadata : object
            Any value to throw between the tensor operations.
        data : arraylike
            The torch tensor or the numpy array to decorate.
            It can be a native python container like a list or a tuple but it is slowler.

        Notes
        -----
        * No range verifications are performed for performance reason.
        * Try to avoid copy if possible, take care of the shared underground data.

        Examples
        --------
        >>> import numpy as np
        >>> import torch
        >>> from laueimproc.classes.image import Image
        >>> arr_numpy = np.ones((5, 5), dtype=np.uint8)
        >>> arr_torch = torch.ones((5, 5), dtype=torch.int16)
        >>> Image(arr_numpy)
        Image([[0.0039, 0.0039, 0.0039, 0.0039, 0.0039],
               [0.0039, 0.0039, 0.0039, 0.0039, 0.0039],
               [0.0039, 0.0039, 0.0039, 0.0039, 0.0039],
               [0.0039, 0.0039, 0.0039, 0.0039, 0.0039],
               [0.0039, 0.0039, 0.0039, 0.0039, 0.0039]])
        >>> Image(arr_torch)
        >>> arr_numpy = arr_numpy.astype(np.float64)
        >>> arr_torch = arr_torch.to(torch.float16)
        >>> arr_torch.requires_grad = True
        >>> Image(arr_numpy)
        Image([[1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.]], dtype=torch.float64)
        >>> _[0, 0] = 0.0  # share underground data
        >>> arr_numpy
        array([[0., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.]])
        >>> Image(arr_torch)  # shared tensor with the same data and properties
        Image([[1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.]], dtype=torch.float16, grad_fn=<AliasBackward0>)
        >>> Image(arr_numpy[1:-1, 1:-1]) # create a view of the data
        Image([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]], dtype=torch.float64)
        >>> _[0, 0] = 0.5 # change the shared underground data
        >>> arr_numpy
        array([[0. , 1. , 1. , 1. , 1. ],
               [1. , 0.5, 1. , 1. , 1. ],
               [1. , 1. , 1. , 1. , 1. ],
               [1. , 1. , 1. , 1. , 1. ],
               [1. , 1. , 1. , 1. , 1. ]])
        >>>
        """
        if isinstance(data, torch.Tensor):
            if not data.dtype.is_floating_point:
                iinfo = torch.iinfo(data.dtype)
                data = data.to(dtype=torch.float32)
                data -= float(iinfo.min)
                data *= 1.0 / float(iinfo.max - iinfo.min)
            image = super().__new__(cls, data) # no copy
            image.metadata = metadata
            return image
        if isinstance(data, np.ndarray):
            if not np.issubdtype(data.dtype, np.floating):
                iinfo = np.iinfo(data.dtype)
                data = data.astype(np.float32)
                data -= float(iinfo.min)
                data *= 1.0 / float(iinfo.max - iinfo.min)
            return Image.__new__(cls, torch.from_numpy(data), metadata=metadata) # no copy
        return Image.__new__(cls, torch.tensor(data), metadata=metadata) # copy

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Enable to throw `metadata` into the new generations.

        Examples
        --------
        >>> import numpy as np
        >>> import torch
        >>> from laueimproc.classes.image import Image
        >>>
        >>> # transmission metadata
        >>> (img := Image([.5], metadata="matadata_value")).metadata
        'matadata_value'
        >>> img.clone().metadata  # deep copy
        'matadata_value'
        >>> torch.sin(img).metadata  # external call
        'matadata_value'
        >>> (img / 2).metadata  # internal method
        'matadata_value'
        >>> img *= 2  # inplace
        >>> img.metadata
        'matadata_value'
        >>> np.sin(img).metadata  # numpy external call
        'matadata_value'
        >>>
        """
        if kwargs is None:
            kwargs = {}
        result = super().__torch_function__(func, types, args, kwargs)
        print("result", result)
        if isinstance(result, cls):
            print("result is subclass of self")
            if isinstance(args[0], cls):  # args[0] is self
                print("add metadata")
                result.metadata = args[0].metadata  # args[0] is self
            else:
                return torch.Tensor(result)
        return result
