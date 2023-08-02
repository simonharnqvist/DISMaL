import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("dismal/segregating_sites.pyx"),
    include_dirs=[numpy.get_include()]
)