#!/usr/bin/env python3

"""Entry point of the cli interface."""

import subprocess
import sys

import click

from laueimproc.common import get_project_root
from laueimproc.io.__main__ import main as main_convert
from laueimproc.testing.__main__ import main as main_test


@click.command()
def main_doc():
    """Try to display the documentation in firefox."""
    index = (get_project_root().parent / "doc" / "build" / "html" / "index.html").as_uri()
    click.echo(index)
    try:
        subprocess.run(["firefox", index], capture_output=True, check=True)
    except FileNotFoundError:
        click.echo(f"failed to open {index} in firefox, paste it in your navigator")


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--version", is_flag=True, help="Display the version of laueimproc.")
def main(ctx=None, version: bool = False):
    """Run the main ``laueimproc`` command line interface."""
    if version:
        from laueimproc import __version__
        click.echo(__version__)
    elif not ctx.invoked_subcommand:
        from laueimproc import __version__
        title = f"* laueimproc {__version__} *"
        click.echo(
            f"{'*'*len(title)}\n{title}\n{'*'*len(title)}\n\n"
            f"Type `{sys.executable} -m laueimproc doc` to read the documentation.\n"
            f"Type `{sys.executable} -m laueimproc --help` to see all the cli options.\n"
        )


main.add_command(main_convert, "convert")
main.add_command(main_doc, "doc")
main.add_command(main_test, "test")


if __name__ == "__main__":
    main()
