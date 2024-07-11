Developer's Guide
=================


Install system libraries
------------------------

To build the documentation, you need ``ffmpeg >= 5.1.5``. You should install it, please refer to the `FFmpeg download page <https://ffmpeg.org/download.html>`_.

.. tab:: Linux

    .. tab:: Debian Ubuntu Mint

        .. code-block:: shell

            sudo apt install ffmpeg
            sudo apt install python-dev pkg-config
            sudo apt install libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev

    .. tab:: RHEL CentOS Fedora

        .. code-block:: shell

            sudo yum install ffmpeg

    .. tab:: Arch Manjaro

        .. code-block:: shell

            sudo pacman -S ffmpeg

    .. tab:: OpenSUSE

        .. code-block:: shell

            sudo zypper install ffmpeg

.. tab:: FreeBSD

    .. code-block:: shell

        sudo pkg install ffmpeg

.. tab:: macOS

    You could install the FFmpeg by using `Homebrew <https://brew.sh/>`_.

    .. code-block:: shell

        brew update
        brew upgrade
        brew install ffmpeg pkg-config

.. tab:: Windows

    It is important to configure your environement variable to hook ffmpeg to LaueImProc. You can follow `this guide <https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/>`_ for example.

To build the documentation, you need ``manim`` as well. You should install it, please refer to the `manim installation page <https://docs.manim.community/en/stable/installation.html>`_.

.. tab:: Linux

    .. tab:: Debian Ubuntu Mint

        .. code-block:: shell

            sudo apt install build-essential python3-dev libcairo2-dev libpango1.0-dev
            sudo apt install texlive texlive-latex-extra

    .. tab:: RHEL CentOS Fedora

        .. code-block:: shell

            sudo dnf install python3-devel
            sudo dnf install cairo-devel pango-devel
            sudo dnf install texlive-scheme-full

    .. tab:: Arch Manjaro

        .. code-block:: shell

            sudo pacman -Syu
            sudo pacman -S cairo pango

.. tab:: macOS

    .. code-block:: shell

        brew install py3cairo
        brew install pango pkg-config
        brew install --cask mactex-no-gui

.. code-block:: shell

    pip install manim


Install all dependencies
------------------------

In developer mode, more dependencies are required. You can install them by adding the ``[all]`` option to ``pip``.

.. warning::
    Make sure you are in a virtual environement ``pyenv activate laueenv`` before excecuting the next lines!

.. code-block:: shell

    cd ~/laueimproc_git/
    pip -v install --editable .[all]


Building Documentation
----------------------

You can also compile documentation locally (after the previous step).

.. code-block:: shell

    cd ~/laueimproc_git/doc/ && make clean && make html && cd -
    laueimproc doc


Test Bench
----------

.. warning::
    Before pushing your changes to GitHub, make sure the test bench is working.

You must also complete the test bench by adding the tests corresponding to your contribution.

.. code-block:: shell

    laueimproc test  # `laueimproc test --help` to see how to skip some tests


Trash
-----

If it segfault, maybe the problem comes from c-files, you can delete it with ``find laueimproc/ -name *.so -exec rm {} \;``.
