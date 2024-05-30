*******************
Laue Pre-Processing
*******************

.. image:: https://img.shields.io/badge/License-MIT-green.svg
    :alt: [license MIT]
    :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/badge/linting-pylint-green
    :alt: [linting: pylint]
    :target: https://github.com/pylint-dev/pylint

.. image:: https://img.shields.io/badge/tests-pass-green
    :alt: [testing]
    :target: https://docs.pytest.org/

.. image:: https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue
    :alt: [versions]
    :target: https://github.com/BM32ESRF/laueimproc/laueimproc/testing


Description
===========

This **generic Laue diagram analyser** ables you to qualify quikly the quality of diferents elements in a big set of Laue diagram brut pictures.
Using **general image processing** tools, it classifies and gives some scores to the spots and the diagrams. The aim is to **didgest very quikely** the big amount of data in order to **help you to select the best subset** for the indexation.


Installation
============

Refer to the ``installation`` tab in the documentation.

To access a pre-built documentation, clone the repository then open the ``index.html`` file with a local browser:

.. code-block:: shell

    if ! [ -d ~/laueimproc_git ]; then git clone https://github.com/BM32ESRF/laueimproc.git ~/laueimproc_git; fi
    firefox ~/laueimproc_git/doc/build/html/index.html


Example
=======

There are a lot of jupyter-notebook examples in the folder ``notebooks`` and a lot of atomic example are directly written in the docstrings.

.. code-block:: python

    import matplotlib.pyplot as plt
    from laueimproc import Diagram, DiagramsDataset
    from laueimproc.io import get_samples

    def init(diag: Diagram) -> int:
        """Find the spots and sorted it by intensities."""
        diag.find_spots()  # peaks search
        diag.filter_spots(diag.compute_rois_sum().argsort(descending=True))  # sorted
        return len(diag)  # nbr of peaks

    diagrams = DiagramsDataset(get_samples())  # create an ordered diagram dataset
    diagrams[:10].apply(init)  # init the 10 first diagrams

    diagrams[6].plot(plt.figure(layout="tight")); plt.show()

.. image:: https://github.com/BM32ESRF/laueimproc/doc/images/diag_06.avif
    :alt: matplotlib figure of diagram 6
