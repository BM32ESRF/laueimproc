Installation
============

Python and PyTorch Support
--------------------------

LaueImproc supports these versions.

.. csv-table:: Python versions
    :file: supported-python-versions.csv
    :header-rows: 1

.. _Debian Ubuntu Mint System:
.. _RHEL CentOS Fedora System:
.. _Arch Manjaro System:
.. _OpenSUSE System:
.. _Linux Installation:
.. _FreeBSD Installation:
.. _macOS Installation:
.. _Windows Installation:


Dependencies
------------

If you have a GPU, please install CUDA or ROC then follow the `PyTorch installation guide <https://pytorch.org/>`_. Without CUDA or ROC, the software is not able to use the GPU (CPU only).

.. tab:: Linux

    .. tab:: Debian Ubuntu Mint

        .. code::

            sudo apt autoremove cuda* nvidia* --purge
            sudo /usr/bin/nvidia-uninstall
            sudo /usr/local/cuda-X.Y/bin/cuda-uninstall
            sudo apt install build-essential gcc dirmngr ca-certificates software-properties-common apt-transport-https dkms curl -y
            curl -fSsL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor | sudo tee /usr/share/keyrings/nvidia-drivers.gpg > /dev/null 2>&1
            echo 'deb [signed-by=/usr/share/keyrings/nvidia-drivers.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /' | sudo tee /etc/apt/sources.list.d/nvidia-drivers.list
            sudo apt update
            apt search cuda-drivers
            sudo apt install nvidia-driver cuda-drivers-545 cuda
            sudo reboot

    .. tab:: RHEL CentOS Fedora

        .. code::

            sudo yum install cuda

    .. tab:: Arch Manjaro

        .. code::

            sudo pacman -S cuda

    .. tab:: OpenSUSE

        .. code::

            sudo zypper install cuda

.. tab:: FreeBSD

    .. code::

        sudo pkg install cuda

.. tab:: macOS

    You could install the FFmpeg by using `Homebrew <https://brew.sh/>`_.

    .. code::

        brew update
        brew upgrade
        brew install cuda pkg-config

.. tab:: Windows

    .. warning:: Windows is crap, so be prepared for a tedious and buggy installation! You should forget Microchiotte-Windaube and go straight to Linux before you pull out all your hair!

    It is important to configure your environement variable to hook cuda to PyTorch. You can follow `this guide <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>`_ for example.


Virtual Environement
--------------------

It is preferable to install laueimproc in a virtual environment. Please refer to the `pyenv installation guide <https://github.com/pyenv/pyenv>`_. It is possible to use ``python3-venv`` as well.


Installation with pip
---------------------

Basic Installation
^^^^^^^^^^^^^^^^^^

.. note::

    The following instructions will install CutCutCodec with simple support for graphical interface.
    See :ref:`building-from-source` for a complete installation including the documentation and the tests.

To install cutcutcodec using `PyPI <https://pypi.org/project/cutcutcodec/>`_, use ``pip``:

.. code::

    pip install --user cutcutcodec[gui]

.. _building-from-source:

Building From Source
^^^^^^^^^^^^^^^^^^^^

To install the lastest development version from `Framagit <https://framagit.org/robinechuca/cutcutcodec>`_ source, clone cutcutcodec using ``git`` and install it using ``pip``:

.. code::

    git clone https://framagit.org/robinechuca/cutcutcodec.git
    cd cutcutcodec/
    python -m pip install --upgrade pip setuptools wheel
    python -m pip -v install --editable .[all]
    $SHELL  # load all the new env-vars, equivalent to restart the shell
    python -m cutcutcodec test  # to test if the installation is ok

You can also compile documentation locally (after the previous step).

.. code::

    cd doc/ && make clean && make html && cd -
    firefox doc/build/html/index.html


Verification
------------

To check that everything is in order, you can run the test bench.
For running tests, some dependencies are requiered, you can install it passing the option ``[all]`` to ``pip``.

.. code::

    python -m cutcutcodec test  # or cutcutcodec-test

To run a partial test, please refer to the integrated CLI:

.. code::

    python -m cutcutcodec test --help


Platform Support
----------------

The tests were successful for teses configurations.

.. note::

    Contributors please test CutCutCodec on your platform then update this document and send a pull request.

+----------------------------------+------------------------+------------------------+-------------------------+
| Operating system                 | Tested Python versions | Tested FFmpeg versions | Tested architecture     |
+==================================+========================+========================+=========================+
| Linux Mint 21.3                  | 3.9, 3.10, 3.11        | 4.4.2                  | x86-64                  |
+----------------------------------+------------------------+------------------------+-------------------------+
