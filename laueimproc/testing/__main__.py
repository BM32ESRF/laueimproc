#!/usr/bin/env python3

"""Alias for ``laueimproc.testing.run``."""

import sys

import click


@click.command()
@click.option("--debug", is_flag=True, help="Show more information on failure.")
@click.option("--skip-install", is_flag=True, help="Do not check the installation.")
@click.option("--skip-coding-style", is_flag=True, help="Do not check programation style.")
@click.option("--skip-slow", is_flag=True, help="Do not run the slow tests.")
def main(debug: bool = False, **kwargs) -> int:
    """Run several tests, alias to ``laueimproc-test``."""
    # no global import for laueimproc.__main__
    from laueimproc.testing.run import run_tests  # pylint: disable=C0415
    return run_tests(
        debug=debug,
        skip_install=kwargs.get("skip_install", False),
        skip_coding_style=kwargs.get("skip_coding_style", False),
        skip_slow=kwargs.get("skip_slow", False),
    )


if __name__ == "__main__":
    sys.exit(main())
