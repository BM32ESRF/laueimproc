#!/usr/bin/env python3

"""The rules of compilation for setuptools."""

from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py

import numpy as np


COMP_RULES = {
    "define_macros": [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],  # for warning
    "extra_compile_args": [
        "-fopenmp",  # for threads
        "-fopenmp-simd",  # for single instruction multiple data
        "-lm",  # for math functions
        "-march=native",  # uses local processor instructions for optimization
        "-O3",  # hight optimization, -O3 include -ffast-math
    ],
    "include_dirs": [np.get_include()],  # requires for  #include numpy
}


class Build(_build_py):
    """Builder to compile c files."""

    def run(self):
        self.run_command("build_ext")  # catch compilation failed ici!
        return super().run()

    def initialize_options(self):
        super().initialize_options()
        if self.distribution.ext_modules is None:
            self.distribution.ext_modules = []
        self.distribution.ext_modules.append(
            Extension(
                "laueimproc.opti.c_rois",
                sources=["laueimproc/opti/c_rois.c"],
                optional=True,
                **COMP_RULES,
            )
        )
        self.distribution.ext_modules.append(
            Extension(
                "laueimproc.improc.spot.c_pca",
                sources=["laueimproc/improc/spot/c_pca.c"],
                optional=True,
                **COMP_RULES,
            )
        )

