#!/usr/bin/env python3

"""Entry point of the cli interface."""

import sys

import click

from laueimproc import __version__
from laueimproc.common import get_project_root
from laueimproc.io.__main__ import main as main_convert
from laueimproc.testing.__main__ import main as main_test


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--version", is_flag=True, help="Display the version of laueimproc.")
def main(ctx=None, version: bool = False):
    """Run the main ``laueimproc`` command line interface."""
    if version:
        click.echo(__version__)
    elif not ctx.invoked_subcommand:
        readme = get_project_root().parent / "README.rst"
        title = f"* laueimproc {__version__} *"
        click.echo(
            f"{'*'*len(title)}\n{title}\n{'*'*len(title)}\n\n"
            f"For a detailed description, please refer to {readme} "
            "or https://github.com/BM32ESRF/laueimproc.\n"
            f"Type `{sys.executable} -m laueimproc --help` to see all the cli options.\n"
        )


main.add_command(main_test, "test")
main.add_command(main_convert, "convert")


if __name__ == "__main__":
    main()
