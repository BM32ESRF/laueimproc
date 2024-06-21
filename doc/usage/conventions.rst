Conventions
===========

Pixel Coordinates
-----------------

The origin is set at the corner of a pixel as shown in the figure below for a simple 4×4 pixel matrix.

.. image:: /build/media/IMGConvIJXY.avif
    :alt: image convention `"ij"` and `"xy"`
    :width: 512

* Each pixel n starts at the coordinated n (included) and goes to the coordinate n+1 (excluded). The center of any pixel is at half integer pixel coordinate.
* The default convention is called `"ij"` and the `LaueTools convention <https://lauetools.readthedocs.io/en/latest/conventions.html>_` is called `"xy"`.
* In the `"ij"` convention, the orgin (i=0, j=0) corresponds to the top left corner of the to left pixel of the image.
* In the `"xy"` convention, the orgin (x=1, y=1) corresponds to the center of the to left pixel of the image.

Motivation
^^^^^^^^^^

.. note::

    The convention `"ij"` is the extension by continuity of the numpy and torch convention, as shown in the below animation.

.. video:: /build/media/ANIMConvIJNumpyContinuity.webm
    :alt: continuity numpy to `"ij"`
    :width: 512

Convertion `"ij"` <=> `"xy"`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most of the proposed methods support the `conv` argument:

.. code-block:: python

    import laueimproc

    diagram = laueimproc.Diagam(laueimproc.io.get_sample())
    diagram.find_spots()
    pos_ij = diagram.compute_rois_centroid(conv="ij")
    pos_xy = diagram.compute_rois_centroid(conv="xy")

In a more general case, you can use the function `ij_to_xy <../../laueimproc/convention.html#laueimproc.convention.ij_to_xy>`_.


Crystalography
--------------

All functions are available in the `laueimproc.diffraction <../../laueimproc/diffraction.html>`_ module.

Lattice parameters
^^^^^^^^^^^^^^^^^^

As shown in the figure below, the parameters a, b, c, alpha, gamma and beta are such that:

* $<e_1, e_2> = cos(\alpha)$
* $<e_1, e_3> = cos(\beta)$
* $<e_1, e_2> = cos(\gamma)$

.. image:: /build/media/IMGLattice.avif
    :alt: lattice parameters
    :width: 512

Projection in cristal base
--------------------------

Here is the convention adopted for projecting the primitive vectors $e_1$, $e_2$ and $e_3$
into the orthonormal base of the christal $C_1$, $C_2$ and $C_3$.
The matrix of the primitive vectors is called $\mathbf{A}$ and the base $\mathcal{B^c}$.

.. video:: /build/media/ANIMLatticeBc.webm
    :alt: projection of lattice in cristal base
    :width: 512

* $e_1$ is collinear with $C_1$.
* $e_2$ in in the plane $C_1, C_2$.
