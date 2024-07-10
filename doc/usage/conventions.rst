Conventions
===========

Pixel Coordinates
-----------------

The origin is set at the corner of a pixel as shown in the figure below for a simple 4×4 pixel matrix.

.. image:: /build/media/IMGConvIJXY.avif
    :alt: image convention `"ij"` and `"xy"`
    :width: 512px

* Each pixel n starts at the coordinated n (included) and goes to the coordinate n+1 (excluded). The center of any pixel is at half integer pixel coordinate.
* The default convention is called `"ij"` and the `LaueTools convention <https://lauetools.readthedocs.io/en/latest/conventions.html>`_ is called `"xy"`.
* In the `"ij"` convention, the orgin (i=0, j=0) corresponds to the top left corner of the to left pixel of the image.
* In the `"xy"` convention, the orgin (x=1, y=1) corresponds to the center of the to left pixel of the image.

Motivation
^^^^^^^^^^

.. note::

    The convention `"ij"` is the extension by continuity of the numpy and torch convention, as shown in the below animation.

.. video:: /build/media/ANIMConvIJNumpyContinuity.webm
    :alt: continuity numpy to `"ij"`
    :width: 512px

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


Geometry
--------

This module follow the `PyFai <https://www.silx.org/doc/pyFAI/latest/geometry.html#detector-position>`_ convention.

It means that the incoming ray is along L3.

.. image:: /build/media/IMGBragg.avif
    :alt: image convention `"ij"` and `"xy"`
    :width: 512px


Crystalography
--------------

All functions are available in the `laueimproc.geometry <../../laueimproc/geometry.html>`_ module.

Lattice Parameters
^^^^^^^^^^^^^^^^^^

As shown in the figure below, the parameters a, b, c, alpha, gamma and beta are such that:

* alpha is the anglee between e2 and e3
* beta is the anglee between e1 and e3
* gamma is the anglee between e1 and e2
* e1 is collinear with C1
* e2 in the plane C1 C2

.. video:: /build/media/ANIMLatticeBc.webm
    :alt: projection of lattice in crystal base
    :width: 512
    :autoplay:
    :loop:

Reciprocal Space
^^^^^^^^^^^^^^^^

Wheter in primitive or reciprocal space, the crystalline mesh is representetd by a 3 x 3 matrix, the concatenation of 3 colums vectors.
The information carried by these matricies are invariant by rotation. In other words, you can express it in any orthonormal base.

Let see how works the reciprocal transformation by the example bellow:

.. video:: /build/media/ANIMPrimitiveReciprocal.webm
    :alt: primitive to reciprocal transformation
    :width: 512
    :autoplay:
    :loop:
