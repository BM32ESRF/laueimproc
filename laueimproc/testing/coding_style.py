#!/usr/bin/env python3

"""Execute tools to estimate the coding style quiality."""

try:
    from pylama.main import shell
except ImportError as err:
    raise ImportError("pylama paquage required (pip install laueimproc[all])") from err

from laueimproc.common import get_project_root


def test_mccabe_pycodestyle_pydocstyle_pyflakes():
    """Run these linters throw pylama on laueimproc."""
    root = get_project_root()
    assert not shell(  # fast checks
        [
            "--options", str(root.parent / "pyproject.toml"),
            "--linters", "mccabe,pycodestyle,pydocstyle,pyflakes", "--async",
            str(root),
        ],
        error=False,
    )


def test_pylint():
    """Run pylint throw pylama on laueimproc."""
    root = get_project_root()
    assert not shell(  # fast checks
        [
            "--options", str(root.parent / "pyproject.toml"),
            "--linters", "pylint",
            str(root),
        ],
        error=False,
    )
