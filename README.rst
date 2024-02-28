*******************
Laue Pre-Processing
*******************

.. image:: https://img.shields.io/badge/License-MIT-green.svg
    :alt: [license MIT]
    :target: https://opensource.org/licenses/MIT


Description
===========

This **generic Laue diagram analyser** ables you to qualify quikly the quality of diferents elements in a big set of Laue diagram brut pictures.
Using **general image processing** tools, it classifies and gives some scores to the spots and the diagrams. The aim is to **didgest very quikely** the big amount of data in order to **help you to select the best subset** for the indexation.


Installation
============


Dependencies
------------

If you have a GPU, please install CUDA or ROC then follow the `PyTorch installation guide <https://pytorch.org/>`_. Without CUDA or ROC, ``laueimproc`` is not able to use the GPU (CPU only).


Installation of the lastest version
-----------------------------------

To install ``laueimproc`` from `GitHub <https://github.com/BM32ESRF/laueimproc>`_ source, clone laueimproc using ``git`` and install it using ``pip`` of ``python3``:

.. code:: bash

    git clone https://github.com/BM32ESRF/laueimproc.git
    cd laueimproc/
    python -m pip install --upgrade pip setuptools wheel
    python -m pip -v install --editable .[all]


Example
=======


.. code:: python

    from laueimproc import Diagram
    
