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


Dependencies (Optional)
-----------------------

If you have a GPU, please install CUDA or ROC then follow the `PyTorch installation guide <https://pytorch.org/>`_. Without CUDA or ROC, the software is not able to use the GPU (CPU only).

.. tab:: Linux

    Follow the official `NVIDIA CUDA Installation Guide for Linux <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_.

.. tab:: FreeBSD

    It is not well supproted, you can try to follow `this bsd link <https://gist.github.com/Mostly-BSD/4d3cacc0ee2f045ed8505005fd664c6e>`_.

.. tab:: macOS

    Follow the `NVIDIA CUDA Installation Guide for Mac OS X <https://docs.nvidia.com/cuda/archive/10.2/cuda-installation-guide-mac-os-x/index.html>`_.

.. tab:: Windows

    .. warning:: Windows is crap, so be prepared for a tedious and buggy installation! You should forget Microchiotte-Windaube and go straight to Linux before you pull out all your hair!

    It is important to configure your environement variable to hook cuda to PyTorch.
    Good luck with the `official guide <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>`_.

As part of laueimproc is written in C, **gcc** must be installed. You can test the correct installation of gcc with the  ``gcc --version`` command.

.. tab:: Linux

    It is installed by default on many linux systems.

    .. tab:: Debian Ubuntu Mint System

        .. code-block:: shell

            sudo apt update
            sudo apt install build-essential
            sudo apt-get install manpages-dev

    .. tab:: RHEL CentOS Fedora System

        .. code-block:: shell

            sudo yum group install "Development Tools"
            sudo yum install man-pages

    .. tab:: Arch Manjaro System

        .. code-block:: shell

            sudo pacman -Syu
            sudo pacman -S base-devel
            sudo pacman -S gcc-multilib lib32-gcc-libs
            wget https://ftp.gnu.org/gnu/gcc/gcc-13.2.0/gcc-13.2.0.tar.gz
            tar -xzvf gcc-13.2.0.tar.gz
            cd gcc-13.2.0
            mkdir build
            cd build
            \../configure
            make -j4
            sudo make install

    .. tab:: OpenSUSE System

        .. code-block:: shell

            sudo zypper refresh
            sudo zypper update
            sudo zypper addrepo http://download.opensuse.org/distribution/leap/15.6/repo/oss/ oss
            zypper search gcc
            sudo zypper install gcc
            sudo zypper install gcc-c++

.. tab:: FreeBSD

    It is install by default on FreeBSD.

.. tab:: macOS

    You could install gcc by using `Homebrew <https://brew.sh/>`_.

    .. code-block:: shell

        brew install gcc

.. tab:: Windows

    .. warning:: I see that you insist on using Windows, this step is the most critical, good luck! Without gcc, some functions of laueimproc will be around **1000 times slowler**.

    It is not too late to listen the voice of reason! You can `install ubuntu <https://lecrabeinfo.net/installer-ubuntu-22-04-lts-le-guide-complet.html>`_ for example.


Virtual Environement
--------------------

It is preferable to install laueimproc in a virtual environment. Please refer to the `pyenv main page <https://github.com/pyenv/pyenv>`_. It is possible to use ``python3-venv`` or ``conda`` as well.

Install pyenv
^^^^^^^^^^^^^

First install the `python dependencies <https://github.com/pyenv/pyenv/wiki#suggested-build-environment>`_ then install pyenv.

.. tab:: Linux

    .. tab:: Debian Ubuntu Mint System

        .. code-block:: shell

            sudo apt update
            sudo apt install libedit-dev libncurses5-dev
            sudo apt install build-essential libssl-dev zlib1g-dev \
            libbz2-dev libreadline-dev libsqlite3-dev curl git \
            libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

    .. tab:: RHEL CentOS Fedora System

        .. code-block:: shell

            sudo yum install openssl11-devel --allowerasing
            yum install gcc make patch zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel tk-devel libffi-devel xz-devel

    .. tab:: Arch Manjaro System

        .. code-block:: shell

            yay -S ncurses5-compat-libs

    .. code-block:: shell

        curl https://pyenv.run | bash
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        source ~/.bashrc

.. tab:: macOS

    You could install dependencies by using `Homebrew <https://brew.sh/>`_.

    .. code-block:: shell

        brew install ncurses
        brew install openssl readline sqlite3 xz zlib tcl-tk
        brew install pyenv

If it fails, `this debug link <https://github.com/pyenv/pyenv/wiki/Common-build-problems>`_ may help you.

Configure pyenv
^^^^^^^^^^^^^^^

Create the virtual environement.

.. code-block:: shell

    pyenv update
    pyenv env PYTHON_CONFIGURE_OPTS="--enable-shared --enable-optimizations --with-lto" PYTHON_CFLAGS='-march=native -mtune=native' install -v 3.12
    pyenv virtualenv 3.12 laueenv
    pyenv activate laueenv

Install jupyter notebook

.. code-block:: shell

    pip install ipython jupyter notebook
    pip install ipympl  # for matplotlib
    # jupyter-notebook


Installation with pip
---------------------

Building From Source
^^^^^^^^^^^^^^^^^^^^

To install the lastest development version from `GitHub <https://github.com/BM32ESRF/laueimproc>`_ source, clone laueimproc using ``git`` and install it using ``pip``:

.. warning::
    Make shure you are in a virtual environement ``pyenv activate laueenv`` before excecuting the next lines!

.. note::
    It works for updating an already installed version as well.

.. code-block:: shell

    if ! [ -d ~/laueimproc_git ]
    then  # download source code
        git clone https://github.com/BM32ESRF/laueimproc.git ~/laueimproc_git
        cd ~/laueimproc_git/
    else  # update source code
        cd ~/laueimproc_git/
        git pull
    fi
    # pyenv activate laueenv  # be sure to be in a virtual env
    pip install --upgrade pip setuptools wheel
    pip -v install --editable .[all]  # compilation and linkage

Building Documentation
^^^^^^^^^^^^^^^^^^^^^^

You can also compile documentation locally (after the previous step).

.. code-block:: shell

    cd ~/laueimproc_git/doc/ && make clean && make html && cd -
    firefox ~/laueimproc_git/doc/build/html/index.html &


Verification
------------

To check that everything is in order, you can run the test bench.
For running tests, some dependencies are requiered, you can install it passing the option ``[all]`` to ``pip``.

.. code-block:: shell

    laueimproc test  # `laueimproc test --help` to see how to skip some tests

If it segfault, maybe the problem comes from c-files, you can delete it with ``find laueimproc/ -name *.so -exec rm {} \;``.


Platform Support
----------------

The tests were successful for teses configurations.

.. note::

    Contributors please test LaueImProc on your platform then update this document and send a pull request.

+----------------------------------+------------------------+-------------------------+
| Operating system                 | Tested Python versions | Tested architecture     |
+==================================+========================+=========================+
| Linux Mint 21.3                  | 3.11                   | x86-64                  |
+----------------------------------+------------------------+-------------------------+
| Ubuntu 22.04                     | 3.12                   | x86-64 13th gen core i7 |
+----------------------------------+------------------------+-------------------------+
