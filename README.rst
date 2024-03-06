*******************
Laue Pre-Processing
*******************

.. image:: https://img.shields.io/badge/License-MIT-green.svg
    :alt: [license MIT]
    :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue
    :alt: [versions]


Description
===========

This **generic Laue diagram analyser** ables you to qualify quikly the quality of diferents elements in a big set of Laue diagram brut pictures.
Using **general image processing** tools, it classifies and gives some scores to the spots and the diagrams. The aim is to **didgest very quikely** the big amount of data in order to **help you to select the best subset** for the indexation.


Installation
============


It is preferable to install LaueImProc in a virtual environment. Please refer to the `pyenv installation guide <https://github.com/pyenv/pyenv>`_. It is possible to use ``python3-venv`` or ``conda`` as well.

Dependencies
------------

If you have a GPU, please install CUDA or ROC then follow the `PyTorch installation guide <https://pytorch.org/>`_. Without CUDA or ROC, the module is not able to use the GPU (CPU only).


Basic Installation
^^^^^^^^^^^^^^^^^^

.. note::

    The project is not yet on pypi! Please install it from source.


Building From Source
^^^^^^^^^^^^^^^^^^^^

To install the lastest development version from `GitHub <https://github.com/BM32ESRF/laueimproc>`_ source, clone LaueImProc using ``git`` and install it using ``pip``:

.. code::

    git clone https://github.com/BM32ESRF/laueimproc.git
    cd laueimproc/
    pip install --upgrade pip setuptools wheel
    pip -v install --editable .[all]

Then, you can generate the documentation with sphinx:

.. code::

    cd doc/
    make clean && make html
    firefox build/html/index.html
    cd -

Or with pdoc3:

.. code::

    pip install pdoc3
    python -m pdoc --force --config latex_math=True --html --output-dir doc/build/pdoc laueimproc
    firefox doc/build/pdoc/laueimproc/index.html


Example
=======

There are a lot of jupyter-notebook examples in the folder ``notebooks``.

.. code:: python

    import matplotlib.pyplot as plt
    import torch
    from laueimproc.io.download import get_samples
    from laueimproc import Diagram

    files = list(get_samples().iterdir())
    diagrams = [Diagram(f) for f in files]
    for diagram in diagrams:
        diagram.find_spots()
    for diagram in diagrams:
        intensities = diagram.compute_pxl_intensities()
        if intensities.shape[0]:
            indexs = torch.argsort(intensities, descending=True)[:10]
            diagram.filter_spots(indexs)
        print(diagram)
        diagram.plot(plt.figure(layout="tight")); plt.show()
