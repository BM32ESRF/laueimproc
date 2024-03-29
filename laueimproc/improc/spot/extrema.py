#!/usr/bin/env python3

"""Search the number of extremums in each rois."""

import torch

KERNEL = torch.tensor(
    [
        [[-1, -1, -1], [1, 1, 1], [0, 0, 0]],  # detect pos top diff h axis
        [[0, 0, 0], [1, 1, 1], [-1, -1, -1]],  # detect neg bottom diff h axis
        [[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]],  # detect pos left diff w axis
        [[0, 1, -1], [0, 1, -1], [0, 1, -1]],  # detect neg right diff w axis
    ],
    dtype=torch.float32,
).reshape(4, 1, 3, 3)


def find_nb_extremums(tensor_spots: torch.Tensor) -> torch.Tensor:
    """Search the number of extremums.

    Parameters
    ----------
    tensor_spots : torch.Tensor
        The batch of spots of shape (n, h, w).

    Returns
    -------
    nbr : torch.Tensor
        The number of extremum in each roi of shape (n,) and dtype int.
    """
    kernel = KERNEL.to(tensor_spots.device)
    cond = torch.nn.functional.conv2d(tensor_spots.unsqueeze(1), kernel) > 0  # (n, 4, h-2, w-2)
    cond = torch.all(cond, dim=1)  # (n, h-2, w-2)
    nb_extremums = torch.sum(cond.to(int), dim=(1, 2))  # (n,)
    return nb_extremums
