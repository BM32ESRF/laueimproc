#!/usr/bin/env python3

"""Affine bilinear batch image interpolation."""

import cv2
import torch

def cv2_inter_nearest(images: torch.Tensor, trans_matrix: torch.Tensor):
    *batch, height, width = images.shape
    images = images.reshape(-1, height, width)  # (n, h, w)
    trans_matrix = trans_matrix.reshape(-1, 2, 3)
    out = torch.cat(
        [
            torch.as_tensor(
                cv2.warpAffine(
                    img,#.numpy(force=True),
                    m,#.numpy(force=True),
                    dsize=(width, height),
                    flags=cv2.INTER_NEAREST,
                    # flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            ).unsqueeze(0)
            for img, m in zip(images.numpy(force=True), trans_matrix.numpy(force=True))
        ]
    )
    out = out.reshape(*batch, height, width)  # (..., h, w)
    return out

# @torch.compile
def inter_nearest(images: torch.Tensor, trans_matrix: torch.Tensor):
    """
    Parameters
    ----------
    images : torch.Tensor
        Of shape (..., h, w).
    trans_matrix : torch.Tensor
        Of shape (..., 2, 3).

    Examples
    --------
    >>> import torch
    >>> from laueimproc.improc.interp import inter_nearest
    >>> images = torch.randn((1000, 30, 40))
    >>> trans_matrix = torch.randn((1000, 2, 3))
    >>> inter_nearest(images, trans_matrix)
    >>>
    """
    *batch, height, width = images.shape
    images = images.reshape(-1, height, width)  # (n, h, w)

    # perfect positions in src images
    i_dst, j_dst = torch.meshgrid(  # not nescessary the same size as input
        torch.arange(0, height, dtype=images.dtype, device=images.device),
        torch.arange(0, width, dtype=images.dtype, device=images.device),
        indexing="ij",
    )  # (h, w)
    dst = torch.cat(  # colum vector [i_dst, j_dst, 1]
        [
            i_dst.reshape(height, width, 1, 1),
            j_dst.reshape(height, width, 1, 1),
            torch.ones((height, width, 1, 1), dtype=images.dtype, device=images.device)
        ],
        axis=2,
    )  # (h, w, 3, 1)
    src = trans_matrix.reshape(-1, 1, 1, 2, 3) @ dst  # (n, h, w, 2, 1)

    # dst = torch.cat(  # colum vector [i_dst, j_dst]
    #     [i_dst.reshape(height, width, 1, 1), j_dst.reshape(height, width, 1, 1)], axis=2
    # )  # (h, w, 2, 1)
    # src = trans_matrix[..., :2].reshape(-1, 1, 1, 2, 2) @ dst  # (n, h, w, 2, 1)
    # src += trans_matrix[..., 2].reshape(-1, 1, 1, 2, 1)

    del dst

    # nearest indexes
    src += 0.5  # transform trunc in round
    indexes = src.to(torch.int32)  # (n, h, w, 2, 1)
    del src
    indexes_i = indexes[..., 0, 0]  # (n, h, w)
    indexes_j = indexes[..., 1, 0]  # (n, h, w)
    indexes_i = torch.clamp(indexes_i, min=0, max=height-1, out=indexes_i)
    indexes_j = torch.clamp(indexes_j, min=0, max=width-1, out=indexes_j)

    # final images
    out = torch.cat(
        [
            images[b][i_l, j_l].unsqueeze(0)
            for b, (i_l, j_l) in enumerate(zip(indexes_i, indexes_j))
        ],
        axis=0,
    )  # (n, h, w)
    out = out.reshape(*batch, height, width)  # (..., h, w)
    return out

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img = cv2.imread("/home/robin/pictures/portrait.jpg", cv2.IMREAD_GRAYSCALE)
    images = torch.as_tensor(img).unsqueeze(0).to(torch.float32)
    import math
    scale = 1.0
    theta = -30 * math.pi/180
    alpha = scale*math.cos(theta)
    beta = scale*math.sin(theta)
    center_i, center_j = images.shape[-2]/2, images.shape[-1]/2
    trans = torch.tensor(
        [[alpha, beta, (1-alpha)*center_i - beta*center_j],
         [-beta, alpha, beta*center_i + (1-alpha)*center_j]]
    ).unsqueeze(0)

    images = images.expand(100, *images.shape[1:])
    trans = trans.expand(100, 2, 3)

    out = inter_nearest(images, trans)
    plt.imshow(out[0])
    plt.show()
    out = cv2_inter_nearest(images, trans)
    plt.imshow(out[0])
    plt.show()
    import timeit
    print(timeit.timeit(lambda: cv2_inter_nearest(images, trans), number=10))
    print(timeit.timeit(lambda: inter_nearest(images, trans), number=10))

