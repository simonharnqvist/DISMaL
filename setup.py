from setuptools import setup

setup(
    name='dismal',
    version='v0.1.15-beta',
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
    ]
)
