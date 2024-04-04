#!/usr/bin/env python3

"""Compare the clustering algos."""

import timeit

import cv2
import numpy as np


def clust_cv2(img: np.ndarray[bool]):
    binary = img.view(np.uint8)
    bboxes = [
        (i, j, h, w) for j, i, w, h in map(  # cv2 to numpy convention
            cv2.boundingRect,
            cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        ) if max(h, w) <= 200 # remove too big spots
    ]

def clust_plantcv(img: np.ndarray[bool]):
    from plantcv import plantcv
    clust_img, clust_mask = plantcv.spatial_clustering(mask=img, algorithm="DBSCAN", min_cluster_size=1)

def bkg():
    import sympy
    i, j, k, l = sympy.symbols("i j k l", real=True)
    # k = l = 0
    i_s, j_s, d_s, l_s = sympy.symbols("i_s j_s d_s l_s", real=True)
    lum = l_s / sympy.sqrt((i_s-i)**2 + (j_s-j)**2 + d_s**2)
    sympy.pprint(lum)
    pkl = sympy.Function("pxl", real=True)(i, j)
    err = sympy.summation(sympy.summation((pkl - lum)**2, (i, 0, k)), (j, 0, l))
    print("err:")
    sympy.pprint(err)
    di_s = err.diff(i_s)
    dj_s = err.diff(j_s)
    dd_s = err.diff(d_s)
    dl_s = err.diff(l_s)
    print("derr/di_s:")
    sympy.pprint(di_s)
    print("derr/dl_s:")
    sympy.pprint(dl_s)


    step = sympy.Symbol("lambda")  # pas
    subs = {i_s: i_s-step*di_s, j_s: j_s-step*dj_s, d_s: d_s-step*dd_s, l_s: l_s-step*dl_s}
    subs = {l_s: l_s-step*dl_s}
    err = sympy.summation(sympy.summation((pkl - lum.xreplace(subs))**2, (i, 0, k)), (j, 0, l))
    print("1d grad direction err:")
    sympy.pprint(err)




if __name__ == "__main__":
    bkg()
    import tqdm
    from laueimproc.io.download import get_samples
    from laueimproc.io.read import read_image
    from laueimproc.improc.peaks_search import estimate_background, _density_to_threshold_numpy, DEFAULT_KERNEL_AGLO
    imgs = []
    for file in tqdm.tqdm(sorted(get_samples().iterdir())):
        img = read_image(file).numpy()
        bkg = estimate_background(img)
        img -= bkg
        threshold = _density_to_threshold_numpy(img, 0.8)
        binary = img > threshold
        binary = cv2.dilate(binary.view(np.uint8), DEFAULT_KERNEL_AGLO, iterations=1)
        # import matplotlib.pyplot as plt
        # plt.imshow(binary)
        # plt.show()
        imgs.append(binary)

    def all_cv2(imgs):
        for img in imgs:
            clust_cv2(img)

    def all_plantcv(imgs):
        for img in imgs:
            clust_plantcv(img)


    t_cv2 = min(timeit.repeat(lambda: all_cv2(imgs), repeat=2, number=2)) / len(imgs)
    print(f"time cv2: {1000*t_cv2:.2f} ms")
    # t_plantcv = min(timeit.repeat(lambda: all_plantcv(imgs), repeat=2, number=2)) / len(imgs)
    # print(f"time plantcv: {1000*t_cv2:.2f} ms")

