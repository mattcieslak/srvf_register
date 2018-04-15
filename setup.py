from __future__ import division, absolute_import, print_function
from glob import glob
import os
import numpy as np
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from glob import glob
import platform

extra_args = ['/openmp'] if platform.system() == "Windows" else ['-fopenmp']

here = os.path.abspath(os.path.dirname(__file__))

srvf_ext = Extension("srvf_register.dynamic_programming_q2",
            sources = ["src/dynamic_programming_q2.pyx", "src/dp_grid.c"],
            extra_compile_args=extra_args,
            extra_link_args=extra_args,
            language="c")

setup(
    name="srvf_register",
    version="0.0",
    packages=["srvf_register"],
    ext_modules = cythonize(srvf_ext,language="c"),
    include_dirs = [np.get_include(),"."]
)


