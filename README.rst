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

If you have a GPU, please install CUDA or ROC then follow the `PyTorch installation guide <https://pytorch.org/>`_. Without CUDA or ROC, ``lauepp`` is not able to use the GPU (CPU only).

It is preferable to install lauepp in a virtual environment. Please refer to the `pyenv installation guide <https://github.com/pyenv/pyenv>`_. It is possible to use ``python3-venv`` as well.


Installation of the lastest version
-----------------------------------

To install ``lauepp`` from `Framagit <https://framagit.org/robinechuca/lauepp>`_ source, clone lauepp using ``git`` and install it using ``pip`` of ``python3``:

.. code:: bash

    git clone https://framagit.org/robinechuca/lauepp.git
    cd lauepp/
    python -m pip install --upgrade pip setuptools wheel
    python -m pip -v install --user --editable .[gui]
    python -m lauepp test


Example
=======


.. code:: python

    from lauepp.io import read_image
