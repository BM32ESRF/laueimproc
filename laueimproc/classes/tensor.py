#!/usr/bin/env python3

"""Define the basic structure of a spot, inerit from torch array."""

import warnings

import numpy as np
import torch



class Tensor(torch.Tensor):
    """A tensor with more options than a torch Tensor.

    Attributes
    ----------
    context : object
        Any information to throw during the operations.
    """

    def __new__(cls, data, context: object = None, to_float: bool = False):
        """Create an image.

        If the image is not in floating point, it is converting in float32
        and normalized beetweeen 0 and 1.

        Parameters
        ----------
        context : object
            Any value to throw between the tensor operations.
            It can be everything, like metadata or contextual informations.
        data : arraylike
            The torch tensor or the numpy array to decorate.
            It can be a native python container like a list or a tuple but it is slowler.
        to_float : boolean, default=False
            If set to True and data are not floating points, it casts the data into
            float and normalize the values between 0 and 1

        Notes
        -----
        * No range verifications are performed for performance reason.
        * Try to avoid copy if possible, take care of the shared underground data.

        Examples
        --------
        >>> import numpy as np
        >>> import torch
        >>> from laueimproc.classes.tensor import Tensor
        >>> arr_numpy = np.ones((5, 5), dtype=np.uint8)
        >>> arr_torch = torch.ones((5, 5), dtype=torch.int16)
        >>> Tensor(arr_numpy)
        Tensor([[0.0039, 0.0039, 0.0039, 0.0039, 0.0039],
               [0.0039, 0.0039, 0.0039, 0.0039, 0.0039],
               [0.0039, 0.0039, 0.0039, 0.0039, 0.0039],
               [0.0039, 0.0039, 0.0039, 0.0039, 0.0039],
               [0.0039, 0.0039, 0.0039, 0.0039, 0.0039]])
        >>> Tensor(arr_torch)
        >>> arr_numpy = arr_numpy.astype(np.float64)
        >>> arr_torch = arr_torch.to(torch.float16)
        >>> arr_torch.requires_grad = True
        >>> Tensor(arr_numpy)
        Tensor([[1., 1., 1., 1., 1.],
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
        >>> Tensor(arr_torch)  # shared tensor with the same data and properties
        Tensor([[1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.]], dtype=torch.float16, grad_fn=<AliasBackward0>)
        >>> Tensor(arr_numpy[1:-1, 1:-1]) # create a view of the data
        Tensor([[1., 1., 1.],
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
        assert isinstance(to_float, bool), to_float.__class__.__name__
        if isinstance(data, torch.Tensor):
            if to_float and not data.dtype.is_floating_point:
                iinfo = torch.iinfo(data.dtype)
                data = data.to(dtype=torch.float32)
                data -= float(iinfo.min)
                data *= 1.0 / float(iinfo.max - iinfo.min)
            image = super().__new__(cls, data) # no copy
            image.context = context
            return image
        if isinstance(data, np.ndarray):
            if to_float and not np.issubdtype(data.dtype, np.floating):
                iinfo = np.iinfo(data.dtype)
                data = data.astype(np.float32)
                data -= float(iinfo.min)
                data *= 1.0 / float(iinfo.max - iinfo.min)
            return Tensor.__new__(cls, torch.from_numpy(data), context=context) # no copy
        warnings.warn(
            "to instanciate a image from a non arraylike data will be forbiden", DeprecationWarning
        )
        return Tensor.__new__(cls, torch.tensor(data), context=context, to_float=to_float) # copy

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Enable to throw `context` into the new generations.

        Examples
        --------
        >>> import numpy as np
        >>> import torch
        >>> from laueimproc.classes.tensor import Tensor
        >>>
        >>> # transmission context
        >>> (img := Tensor([.5], context="matadata")).context
        'matadata'
        >>> img.clone().context  # deep copy
        'matadata'
        >>> torch.sin(img).context  # external call
        'matadata'
        >>> (img / 2).context  # internal method
        'matadata'
        >>> img *= 2  # inplace
        >>> img.context
        'matadata'
        >>> np.sin(img).context  # numpy external call
        'matadata'
        >>>
        """
        if kwargs is None:
            kwargs = {}
        result = super().__torch_function__(func, types, args, kwargs)
        if isinstance(result, cls):
            if isinstance(args[0], cls):  # args[0] is self
                result.context = getattr(args[0], "context", None)  # args[0] is self
            else:
                return torch.Tensor(result)
        return result
