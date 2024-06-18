Developer's Guide
=================


Prerequistes
------------

In developer mode, more dependencies are required. You can install them by adding the ``[all]`` option to ``pip``.

.. warning::
    Make shure you are in a virtual environement ``pyenv activate laueenv`` before excecuting the next lines!

.. code-block:: shell

    cd ~/laueimproc_git/
    pip -v install --editable .[all]

To build the documentation, you need ``ffmpeg >= 5.1.5``. You should install it, please refer to the `FFmpeg download page <https://ffmpeg.org/download.html>`_.

.. tab:: Linux

    .. tab:: Debian Ubuntu Mint

        .. code-block:: shell

            sudo apt update
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


Building Documentation
----------------------

You can also compile documentation locally (after the previous step).

.. code-block:: shell

    cd ~/laueimproc_git/doc/ && make clean && make html && cd -
    laueimproc doc


Test Bench
----------

.. warning::
    Before pushing your changes to GitHub, make shure the test bench is working.

You must also complete the test bench by adding the tests corresponding to your contribution.

.. code-block:: shell

    laueimproc test  # `laueimproc test --help` to see how to skip some tests


Trash
-----

If it segfault, maybe the problem comes from c-files, you can delete it with ``find laueimproc/ -name *.so -exec rm {} \;``.


Pour installer xypython
Les liens https://gist.github.com/oleksis/f897d0b186bcc30b29b6ac7ef65320ed
puis https://askubuntu.com/questions/1330171/how-to-install-wxpython-under-an-alternative-python-version
sont utiles, adapter la commande git à https://extras.wxpython.org/wxPython4/extras/linux/gtk3/
