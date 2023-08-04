import numpy
from setuptools import setup
from Cython.Build import cythonize

from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

setup(
    ext_modules=cythonize("dismal/segregating_sites.pyx"),
    include_dirs=[numpy.get_include()],
)