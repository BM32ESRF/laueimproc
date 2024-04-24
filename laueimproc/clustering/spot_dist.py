#!/usr/bin/env python3

"""Find the close spots in two diagrams."""

import numbers
import typing

import numpy as np
import torch


def associate_spots(
    pos1: torch.Tensor,
    pos2: torch.Tensor,
    eps: numbers.Real,
) -> torch.Tensor:
    """Find the close spots.

    Parameters
    ----------
    pos1 : torch.Tensor
        The coordinates of the position of each spot of the first diagram.
        Is is a tensor of shape (n, 2) containg real elements.
    pos2 : torch.Tensor
        The coordinates of the position of each spot of the second diagram.
        Is is a tensor of shape (n, 2) containg real elements.
    eps : float
        The max euclidian distance to associate 2 spots.

    Returns
    -------
    pair : torch.Tensor
        The couple of the indices of the closest spots, of shape (n, 2).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.clustering.spot_dist import associate_spots
    >>> pos1 = torch.rand((1500, 2), dtype=torch.float32)
    >>> pos2 = torch.rand((2500, 2), dtype=torch.float32)
    >>> eps = 1e-3
    >>> pair = associate_spots(pos1, pos2, eps)
    >>> (torch.sqrt((pos1[pair[:, 0]] - pos2[pair[:, 1]])**2) <= eps).all()
    tensor(True)
    >>>
    """
    assert isinstance(pos1, torch.Tensor), pos1.__class__.__name__
    assert isinstance(pos2, torch.Tensor), pos2.__class__.__name__
    assert pos1.ndim == 2 and pos1.shape[1] == 2, pos1.shape
    assert pos2.ndim == 2 and pos2.shape[1] == 2, pos2.shape
    assert pos1.dtype.is_floating_point, pos1.dtype
    assert pos2.dtype.is_floating_point, pos2.dtype
    assert isinstance(eps, numbers.Real), eps.__class__.__name__
    assert eps >= 0, eps

    full_dist_square = pos1.unsqueeze(1) - pos2.unsqueeze(0)  # broadcast (n1, n2, 2)
    full_dist_square *= full_dist_square
    dist_square = torch.sum(full_dist_square, dim=2)  # euclid_dist[idx1, idx2]**2
    values, idx2 = torch.min(dist_square, dim=1)
    to_keep = values <= eps**2
    idx1 = torch.arange(len(pos1), dtype=idx2.dtype, device=idx2.device)
    pair = torch.cat((idx1[to_keep].unsqueeze(1), idx2[to_keep].unsqueeze(1)), dim=1)
    return pair


def group_spots(
    spot_to_diags: dict[int, set[int]],
    eps: numbers.Real = 0.25,
    min_samples: numbers.Integral = 3,
) -> dict[int, set[int]]:
    """Group the spots sharing the same set of diagrams.

    Parameters
    ----------
    spot_to_diags : dict[int, set[int]]
        To each spot index, associate the set of diagram indices, containg the spot.
    eps : float
        The maximum distance between two set of diagrams to consider they share the same cluster.
    min_samples : int
        The minimum cardinal of a cluster.

    Returns
    -------
    clusters : dict[int, set[int]]
        To each cluster label, associate the set of spots labels.

    Examples
    --------
    >>> from pprint import pprint
    >>> from laueimproc.clustering.spot_dist import group_spots
    >>> spot_to_diag = {
    ...     0: {0, 1, 4},
    ...     1: {0, 1, 4},
    ...     2: {1, 2, 4, 5},
    ...     3: {1, 2, 3, 4, 5},
    ...     4: {2, 3, 4, 5},
    ...     5: {0, 1, 2, 3, 4, 5},
    ...     6: {0, 1, 2, 4, 5},
    ... }
    >>> pprint(group_spots(spot_to_diag, eps=0.17, min_samples=2))
    {0: {0, 1}, 1: {3, 5, 6}}
    >>>
    """
    assert isinstance(eps, numbers.Real), eps.__class__.__name__
    assert 0 <= eps < 1, eps
    assert isinstance(min_samples, numbers.Integral), min_samples
    assert min_samples >= 1, min_samples

    from sklearn.cluster import DBSCAN  # pylint: disable=C0415

    # compute the matrix of distances, using the Jaccard norm
    dist_matrix = np.zeros((len(spot_to_diags), len(spot_to_diags)), dtype=np.float32)
    spot_labels = list(spot_to_diags)
    diag_sets = [spot_to_diags[s] for s in spot_labels]  # not .values() for order
    for i, set_1 in enumerate(diag_sets[:-1]):
        for j, set_2 in zip(range(i+1, len(diag_sets)), diag_sets[i+1:]):
            inter_len = float(len(set_1 & set_2))
            dist_matrix[i, j] = 1.0 - (inter_len / (len(set_1) + len(set_2) - inter_len))
    dist_matrix += dist_matrix.transpose()

    # clustering
    lbls = DBSCAN(
        eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1
    ).fit_predict(dist_matrix)

    # cast result
    clusters = {}
    for i, lbl in enumerate(lbls.tolist()):
        if lbl == -1:
            continue
        clusters[lbl] = clusters.get(lbl, set())
        clusters[lbl].add(spot_labels[i])
    return clusters


def spotslabel_to_diag(labels: dict[int, torch.Tensor]) -> dict[int, set[int]]:
    """Inverse the representation, from diagram to spots.

    Parameters
    ----------
    labels : dict[int, torch.Tensor]
        To each diagram index, associate the label of each spot as a compact dict.
        Each value is a tensor of shape (n, 2), first column is native spot index into the diagram,
        then second column corresponds to the label.

    Returns
    -------
    diagrams : dict[int, set[int]]
        To each spot label, associate the set of diagram indices, containg the spot.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.clustering.spot_dist import (associate_spots, track_spots,
    ...     spotslabel_to_diag)
    >>> h, w = 5, 10
    >>> diags = torch.tensor(
    ...     sum(([[j, j+1] for j in range(i*w, (i+1)*w-1)] for i in range(h)), start=[])
    ...     + sum(([[j, j+w] for j in range(i*w, (i+1)*w)] for i in range(h-1)), start=[])
    ... )
    >>> pos = [torch.rand((2000, 2), dtype=torch.float32) for _ in range(diags.max()+1)]
    >>> pairs = [associate_spots(pos[i], pos[j], 5e-3) for i, j in diags.tolist()]
    >>> labels = track_spots(pairs, diags)
    >>> diagrams = spotslabel_to_diag(labels)
    >>>
    """
    assert isinstance(labels, dict), labels.__class__.__name__
    diagrams = {}
    for diag, spotidx_to_label in labels.items():
        assert isinstance(diag, int), diag.__class__.__name__
        assert isinstance(spotidx_to_label, torch.Tensor), spotidx_to_label.__class__.__name__
        assert spotidx_to_label.ndim == 2 and spotidx_to_label.shape[1] == 2, spotidx_to_label.shape
        for spot_label in spotidx_to_label[:, 1].tolist():
            diagrams[spot_label] = diagrams.get(spot_label, set())
            diagrams[spot_label].add(diag)
    return diagrams


def track_spots(
    pairs: list[torch.Tensor],
    diags: torch.Tensor,
    *, _labels_rename: typing.Optional[tuple[dict[int, torch.Tensor], dict[int, int]]] = None,
) -> dict[int, torch.Tensor]:
    """Associate one label by position, give this label to all spots at this position.

    Parameters
    ----------
    pairs : list[torch.Tensor]
        For each pair of diagrams, contains the couple of close spots indices.
    diags : torch.Tensor
        The index of the diagrams for each pair, shape (len(pairs), 2).

    Returns
    -------
    labels : dict[int, torch.Tensor]
        To each diagram index, associate the label of each spot as a compact dict.
        Each value is a tensor of shape (n, 2), first column is native spot index into the diagram,
        then second column corresponds to the label.

    Examples
    --------
    >>> import torch
    >>> from laueimproc.clustering.spot_dist import associate_spots, track_spots
    >>> h, w = 5, 10
    >>> diags = torch.tensor(
    ...     sum(([[j, j+1] for j in range(i*w, (i+1)*w-1)] for i in range(h)), start=[])
    ...     + sum(([[j, j+w] for j in range(i*w, (i+1)*w)] for i in range(h-1)), start=[])
    ... )
    >>> pos = [torch.rand((2000, 2), dtype=torch.float32) for _ in range(diags.max()+1)]
    >>> pairs = [associate_spots(pos[i], pos[j], 5e-3) for i, j in diags.tolist()]
    >>> labels = track_spots(pairs, diags)
    >>>
    """
    assert isinstance(pairs, list), pairs.__class__.__name__
    assert all(isinstance(p, torch.Tensor) and p.ndim == 2 and p.shape[1] == 2 for p in pairs)
    assert isinstance(diags, torch.Tensor), diags.__class__.__name__
    assert diags.shape == (len(pairs), 2), diags.shape

    if not pairs:
        return {}

    # preparation
    dtype, device = diags.dtype, diags.device
    diags = diags.tolist()
    if _labels_rename is not None:
        labels = {diag_idx: dict(spots.tolist()) for diag_idx, spots in _labels_rename[0].items()}
        next_label = max((int(spots[:, 1].max()) for spots in _labels_rename[0].values()), start=0)
        rename = _labels_rename[1]
    else:
        labels = {diag_idx: {} for indices in diags for diag_idx in indices}
        next_label = 0  # the max label name
        rename = {}  # old name -> new name

    # explore and update graph
    for (diag_idx1, diag_idx2), pair in zip(diags, pairs):
        for spot1, spot2 in pair.tolist():
            if spot1 not in labels[diag_idx1] and spot2 not in labels[diag_idx2]:  # case new spot
                labels[diag_idx1][spot1] = labels[diag_idx2][spot2] = next_label
                next_label += 1
            elif spot1 not in labels[diag_idx1] and spot2 in labels[diag_idx2]:  # link 2 to 1
                labels[diag_idx1][spot1] = labels[diag_idx2][spot2]
            elif spot1 in labels[diag_idx1] and spot2 not in labels[diag_idx2]:  # link 1 to 2
                labels[diag_idx2][spot2] = labels[diag_idx1][spot1]
            elif (name1 := labels[diag_idx1][spot1]) != (name2 := labels[diag_idx2][spot2]):
                rename[name2] = name1

    # rename
    while (new_rename := {s: rename.get(n, n) for s, n in rename.items()}) != rename:
        rename = new_rename  # avoid cyclic reference
    labels = {
        diag_idx: {s: rename.get(n, n) for s, n in spots.items()}
        for diag_idx, spots in labels.items()
    }

    # cast
    labels = {
        diag_idx: torch.tensor(list(spots.items()), dtype=dtype, device=device)
        for diag_idx, spots in labels.items()
    }

    return labels if _labels_rename is None else (labels, rename)
