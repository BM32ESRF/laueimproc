#!/usr/bin/env python3

"""Allow cli for convertion."""

import sys

import click


@click.command()
@click.argument("src", nargs=-1, type=click.Path(exists=True))
def main(src: tuple = ()) -> int:
    """Convert images into lossless jpeg2000 images."""
    from laueimproc.io.convert import to_jp2  # no global import for laueimproc.__main__
    return to_jp2(src, dst_dir=None)


if __name__ == "__main__":
    sys.exit(main())
