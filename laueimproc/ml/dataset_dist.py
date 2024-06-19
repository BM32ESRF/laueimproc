#!/usr/bin/env python3

"""Manage the diagams positioning inside a dataset."""

import inspect
import logging
import numbers
import typing
import warnings

import torch

try:
    from laueimproc.ml import c_dist
except ImportError:
    logging.warning(
        "failed to import laueimproc.ml.c_dist, a slow python version is used instead"
    )
    c_dist = None


SCALS_TYPE = typing.Union[dict[int, tuple[float, ...]], tuple[torch.Tensor, torch.Tensor]]
DIAG2SCALS_TYPE = typing.Callable[[int], typing.Union[numbers.Real, tuple[numbers.Real, ...]]]


def _add_pos(pos: SCALS_TYPE, index: int, coords: tuple[float, ...]) -> SCALS_TYPE:
    """Append a nez position to the positions.

    Notes
    -----
    No check for performance reason.
    """
    pos = _to_dict(pos)
    pos[index] = coords
    return pos


def _copy_pos(pos: SCALS_TYPE) -> SCALS_TYPE:
    """Deep copy of the position.

    Notes
    -----
    No check for performance reason.
    """
    # Convert in tensor because it is more compact in memory
    # and this function is inderectly used for serialization.
    if isinstance(pos, dict):
        pos = _to_torch(pos)
    return (pos[0].clone(), pos[1].clone())


def _filter_pos(pos: SCALS_TYPE, indices: typing.Iterable[int]) -> SCALS_TYPE:
    """Select only some positions.

    Notes
    -----
    No check for performance reason.
    indicies are assumed to be in pos.
    """
    pos = _to_dict(pos)
    pos = {i: pos[i] for i in indices}
    return pos


def _to_dict(pos: SCALS_TYPE) -> dict[int, tuple[float, ...]]:
    """Convert the general position into a dictionary form.

    Notes
    -----
    No check for performance reason.
    """
    if isinstance(pos, dict):
        return pos
    indices, coords = pos
    return {i: c for i, *c in zip(indices.tolist(), coords.tolist())}


def _to_torch(pos: SCALS_TYPE) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert the general position into the torch form.

    Notes
    -----
    No check for performance reason.
    """
    if isinstance(pos, tuple):
        return pos
    indices, coords = zip(*pos.items())
    indices_torch = torch.asarray(indices, dtype=torch.int32)
    coords_torch = torch.asarray(coords, dtype=torch.float32)
    return (indices_torch, coords_torch)


def call_diag2scalars(pos_func: DIAG2SCALS_TYPE, index: int) -> tuple[float, ...]:
    """Call the function, check, cast and return the output.

    The function typing is assumed to be checked.

    Parameters
    ----------
    pos_func : callable
        The function that associate a space position to a diagram index.
    index : int
        The argument value of the function.

    Returns
    -------
    position : tuple[float, ...]
        The scalar vector as a tuple of float.
    """
    out = pos_func(index)
    if isinstance(out, numbers.Real):
        out = (out,)
    else:
        assert isinstance(out, tuple), \
            f"the function {pos_func} has to return a scalar or a tuple of scalars, not {out}"
    for i, item in enumerate(out):
        assert isinstance(item, numbers.Real), f"the coordinate of index {i} is not a scalar {item}"
    out = tuple(map(float, out))
    return out


def check_diag2scalars_typing(pos_func: DIAG2SCALS_TYPE):
    """Ensure that the position function has the right type of input / outputs.

    Parameters
    ----------
    pos_func : callable
        A function supposed to take a diagram index as input
        and that return a scalar vector in a space of n dimensions.

    Raises
    ------
    AssertionError
        If something wrong is detected.

    Examples
    --------
    >>> import pytest
    >>> from laueimproc.ml.dataset_dist import check_diag2scalars_typing
    >>> def ok_1(index: int) -> float:
    ...     return float(index)
    ...
    >>> def ok_2(index: int) -> tuple[float]:
    ...     return (float(index),)
    ...
    >>> def ok_3(index: int) -> tuple[float, float]:
    ...     return (float(index), 0.0)
    ...
    >>> def warn_1(index):
    ...     return float(index)
    ...
    >>> def warn_2(index: int):
    ...     return float(index)
    ...
    >>> def warn_3(index) -> float:
    ...     return float(index)
    ...
    >>> def warn_4(index: int) -> tuple:
    ...     return float(index)
    ...
    >>> error_1 = "this is not a function"
    >>> def error_2(file: str) -> float:  # bad input type
    ...     return float(file)
    ...
    >>> def error_3(index: int) -> list:  # bad output type
    ...     return [float(index)]
    ...
    >>> def error_4(index: int, cam: str) -> tuple:  # bag input arguments
    ...     return float(index)
    ...
    >>> check_diag2scalars_typing(ok_1)
    >>> check_diag2scalars_typing(ok_2)
    >>> check_diag2scalars_typing(ok_3)
    >>>
    >>> with pytest.warns(SyntaxWarning):
    ...     check_diag2scalars_typing(warn_1)
    ...
    >>> with pytest.warns(SyntaxWarning):
    ...     check_diag2scalars_typing(warn_2)
    ...
    >>> with pytest.warns(SyntaxWarning):
    ...     check_diag2scalars_typing(warn_3)
    ...
    >>> with pytest.warns(SyntaxWarning):
    ...     check_diag2scalars_typing(warn_4)
    ...
    >>>
    >>> with pytest.raises(AssertionError):
    ...     check_diag2scalars_typing(error_1)
    ...
    >>> with pytest.raises(AssertionError):
    ...     check_diag2scalars_typing(error_2)
    ...
    >>> with pytest.raises(AssertionError):
    ...     check_diag2scalars_typing(error_3)
    ...
    >>> with pytest.raises(AssertionError):
    ...     check_diag2scalars_typing(error_4)
    ...
    >>>
    """
    assert callable(pos_func), f"{pos_func} has to be callable, not {pos_func.__class__.__name__}"
    signature = inspect.signature(pos_func)
    assert len(signature.parameters) == 1, \
        f"the function {pos_func} has to take exactely 1 parameter, not {signature.parameters}"
    parameter = next(iter(signature.parameters.values()))
    if parameter.annotation is parameter.empty:
        warnings.warn(f"please specify the input type of {pos_func}", SyntaxWarning)
    elif parameter.annotation is not int:
        raise AssertionError(
            f"the function {pos_func} has to get a int as input, not {parameter.annotation}"
        )
    if not (
        inspect.isclass(signature.return_annotation)
        and issubclass(signature.return_annotation, numbers.Real)
    ):
        if signature.return_annotation is parameter.empty:
            warnings.warn(f"please specify the return type of {pos_func}", SyntaxWarning)
            return
        if issubclass(signature.return_annotation, tuple):
            warnings.warn(
                f"please specify the number of scalars returned by {pos_func}", SyntaxWarning
            )
            return
        origin = typing.get_origin(signature.return_annotation)
        assert origin is not None and issubclass(origin, tuple), (
            f"the function {pos_func} has to return a tuple or a scalar real number, "
            f"not {signature.return_annotation}"
        )
        return_args = typing.get_args(signature.return_annotation)
        assert return_args, f"the function {pos_func} has to return at leat one element"
        for i, return_arg in enumerate(return_args):
            assert issubclass(return_arg, numbers.Real), \
                f"returned element {i} of {pos_func} has to be a real number, not a {return_arg}"


def select_closest(
    coords: torch.Tensor,
    point: tuple[float, ...],
    tol: typing.Optional[tuple[float, ...]] = None,
    scale: typing.Optional[tuple[float, ...]] = None,
    *, _no_c: bool = False,
) -> int:
    r"""Select the closest point.

    Find the index i such as \(d_i\) is minimum, using the following formalism:
    \(\begin{cases}
        d_i = \sqrt{\sum\limits_{j=0}^{D-1}\left(\kappa_j(p_j-x_{ij}))^2\right)} \\
        \left|p_j-x_{ij}\right| \le \epsilon_j, \forall j \in [\![0;D-1]\!] \\
    \end{cases}\)

    * \(D\), the number of dimensions of the space used.
    * \(\kappa_j\), a scalar inversely homogeneous has the unit used by the quantity of index \(j\).
    * \(p_j\), the coordinate \(j\) of the point of reference.
    * \(x_{ij}\), the \(i\)-th point of comparaison, coordinate \(j\).

    Parameters
    ----------
    coords : torch.Tensor
        The float32 points of each individual \(\text{coords[i, j]} = x_{ij}\), of shape (n, \(D\)).
    point : tuple[float, ...]
        The point of reference in the destination space \(point[j] = p_j\).
    tol : tuple[float, ...], default inf
        The absolute tolerence value for each component (kind of manhattan distance).
        Such as \(\text{tol[j]} = \epsilon_j\).
    scale : tuple[float, ...], optional
        \(\text{scale[j]} = \kappa_j\),
        used for rescale each axis before to compute the euclidian distance.
        By default \(\kappa_j = 1, \forall j \in [\![0;D-1]\!]\).

    Returns
    -------
    index: int
        The index \(i\) of the closest item \(\underset{i}{\operatorname{argmin}}\left(d\right)\).

    Raises
    ------
    LookupError
        If no points match the criteria.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.ml.dataset_dist import select_closest
    >>> coords = torch.empty((1000, 3), dtype=torch.float32)
    >>> coords[:, 0] = torch.linspace(-1, 1, 1000)
    >>> coords[:, 1] = torch.linspace(-10, 10, 1000)
    >>> coords[:, 2] = torch.arange(1000) % 2
    >>> select_closest(coords, (0.0, 0.0, 0.1))
    500
    >>> select_closest(coords, (0.0, 0.0, 0.9))
    499
    >>> select_closest(coords, (0.5, 5.0, 0.1))
    750
    >>> select_closest(coords, (0.5, 5.0, 0.1), scale=(10, 1, 0.01))
    749
    >>> select_closest(coords, (0.0, 0.0, 0.1), tol=(4/1000, 40/1000, 0.2))
    500
    >>> try:
    ...    select_closest(coords, (0.0, 0.0, 0.1), tol=(1/1000, 10/1000, 0.05))
    ... except LookupError as err:
    ...     print(err)
    ...
    no point match
    >>>
    """
    if not _no_c and c_dist is not None:
        coords_np = coords.numpy(force=True)
        kwargs = ({} if tol is None else {"tol": tol}) | ({} if scale is None else {"scale": scale})
        return c_dist.select_closest_point(coords_np, point, **kwargs)

    assert isinstance(coords, torch.Tensor), coords.__class__.__name__
    assert coords.dtype == torch.float32, coords.dtype
    assert coords.ndim == 2, coords.shape
    assert isinstance(point, tuple), point.__class__.__name__
    assert point, "at least dimension 1, not 0"
    assert all(isinstance(c, numbers.Real) for c in point), point
    assert len(point) == coords.shape[1], (point, coords.shape)
    if tol is not None:
        assert isinstance(tol, tuple), tol.__class__.__name__
        assert all(isinstance(e, numbers.Real) for e in tol), tol
        assert len(tol) == coords.shape[1], (tol, coords.shape)
    if scale is not None:
        assert isinstance(scale, tuple), scale.__class__.__name__
        assert all(isinstance(k, numbers.Real) for k in scale), scale
        assert len(scale) == coords.shape[1], (scale, coords.shape)

    # preparation
    indices = torch.arange(len(coords), dtype=torch.int32, device=coords.device)

    # reject items
    if tol is not None:
        for i, (point_c, tol_c) in enumerate(zip(point, tol)):
            if tol is not None:
                keep = coords[:, i] >= point_c - tol_c
                indices, coords = indices[keep], coords[keep]
                keep = coords[:, i] <= point_c + tol_c
                indices, coords = indices[keep], coords[keep]
    if not indices.shape[0]:
        raise LookupError("no point match")

    # compute dist
    dist = coords - torch.asarray(point, dtype=coords.dtype, device=coords.device).unsqueeze(0)
    if scale is not None:
        dist *= torch.asarray(scale, dtype=coords.dtype, device=coords.device).unsqueeze(0)
    dist *= dist
    dist = torch.sum(dist, dim=1)

    # keep closet
    index = int(torch.argmin(dist))
    index = int(indices[index])
    return index


def select_closests(
    coords: torch.Tensor,
    point: typing.Optional[tuple[float, ...]] = None,
    tol: typing.Optional[tuple[float, ...]] = None,
    scale: typing.Optional[tuple[float, ...]] = None,
    *, _no_c: bool = False,
) -> torch.Tensor:
    r"""Select the closest points.

    Find all the indices i such as:
    \(\left|p_j-x_{ij}\right| \le \epsilon_j, \forall j \in [\![0;D-1]\!]\)

    Sorted the results byt increasing \(d_i\) such as:
    \(d_i = \sqrt{\sum\limits_{j=0}^{D-1}\left(\kappa_j(p_j-x_{ij}))^2\right)}\)

    * \(D\), the number of dimensions of the space used.
    * \(\kappa_j\), a scalar inversely homogeneous has the unit used by the quantity of index \(j\).
    * \(p_j\), the coordinate \(j\) of the point of reference.
    * \(x_{ij}\), the \(i\)-th point of comparaison, coordinate \(j\).

    Parameters
    ----------
    coords : torch.Tensor
        The float32 points of each individual \(\text{coords[i, j]} = x_{ij}\), of shape (n, \(D\)).
    point : tuple[float, ...], optional
        If provided, the point \(point[j] = p_j\) is used to calculate the distance
        and sort the results.
        By default, the point taken is equal to the average of the tol.
    tol : tuple[float | tuple[float, float], ...], default inf
        The absolute tolerence value for each component (kind of manhattan distance).
        Such as \(\text{tol[j][0]} = \epsilon_{j-min}, \text{tol[j][1]} = \epsilon_{j-max}\).
    scale : tuple[float, ...], optional
        \(\text{scale[j]} = \kappa_j\),
        used for rescale each axis before to compute the euclidian distance.
        By default \(\kappa_j = 1, \forall j \in [\![0;D-1]\!]\).

    Returns
    -------
    indices : torch.Tensor
        The int32 list of the sorted coords indices.
    """
    if point is not None:
        assert isinstance(coords, torch.Tensor), coords.__class__.__name__
        assert coords.dtype == torch.float32, coords.dtype
        assert coords.ndim == 2, coords.shape
        assert isinstance(point, tuple), point.__class__.__name__
        assert point, "at least dimension 1, not 0"
        assert all(isinstance(c, numbers.Real) for c in point), point
        assert len(point) == coords.shape[1], (point, coords.shape)
    if tol is not None:
        assert isinstance(tol, tuple), tol.__class__.__name__
        assert (
            all(isinstance(e, numbers.Real) for e in tol)
            or (
                all(isinstance(e, tuple) for e in tol)
                and all(len(e) == 2 for e in tol)
                and all(
                    isinstance(e[0], numbers.Real) and isinstance(e[1], numbers.Real) for e in tol
                )
            )
        ), tol
        assert len(tol) == coords.shape[1], (tol, coords.shape)
    if scale is not None:
        assert isinstance(scale, tuple), scale.__class__.__name__
        assert all(isinstance(k, numbers.Real) for k in scale), scale
        assert len(scale) == coords.shape[1], (scale, coords.shape)
    assert point is not None or (tol is not None and isinstance(tol[0], tuple))

    # preparation
    if point is None and tol is not None and isinstance(tol[0], tuple):
        point = tuple(0.5 * (e[0] + e[1]) for e in tol)
    if tol is not None and isinstance(tol[0], numbers.Real):  # points assumed to be not None
        tol = tuple((p-e, p+e) for p, e in zip(point, tol))
    indices = torch.arange(len(coords), dtype=torch.int32, device=coords.device)

    # reject items
    if tol is not None and len(indices):
        for i, (tol_c_min, tol_c_max) in enumerate(tol):
            keep = coords[:, i] >= tol_c_min
            indices, coords = indices[keep], coords[keep]
            keep = coords[:, i] <= tol_c_max
            indices, coords = indices[keep], coords[keep]

    # compute dist and sorted
    if point is not None:
        dist = coords - torch.asarray(point, dtype=coords.dtype, device=coords.device).unsqueeze(0)
        if scale is not None:
            dist *= torch.asarray(scale, dtype=coords.dtype, device=coords.device).unsqueeze(0)
        dist *= dist
        dist = torch.sum(dist, dim=1)
        indices = indices[torch.argsort(dist)]

    # keep closet
    return indices
