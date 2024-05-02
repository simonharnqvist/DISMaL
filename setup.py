from setuptools import setup

setup(
    name='dismal',
    version='v0.1.16-beta',
    packages=['dismal'],
    install_requires=[
        "numpy",
        "pandas",
        "setuptools",
        "pyranges",
        "scikit-allel",
        "demes",
        "demesdraw",
        "matplotlib",
        "msprime",
        "seaborn",
        "tqdm",
        "tskit",
        "prettytable"
    ],
    long_description=open('README.rst').read(),
)
