from setuptools import setup

setup(
    name='dismal',
    version='v0.1.18-beta',
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
        "prettytable",
        "joblib"
    ],
    description="Demographic inference from the distribution of pairwise segregating sites",
    long_description=open('README.rst').read(),
)
