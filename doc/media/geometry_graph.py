#!/usr/bin/env python3

"""Draw the graph with networkx and graphviz."""

import pathlib

import networkx as nx

from laueimproc.geometry.link import GRAPH


def main():
    """Draw the graph image."""
    print("create graph...")
    agraph = nx.nx_agraph.to_agraph(GRAPH)  # to pygraphviz
    agraph.layout("dot")
    pathlib.Path("/tmp/images").mkdir(parents=True, exist_ok=True)
    agraph.draw("/tmp/images/IMGGeometryGraph.png")
    print("    done")


if __name__ == "__main__":
    main()
