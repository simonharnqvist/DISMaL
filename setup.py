from setuptools import setup

setup(
    name='dismal',
    version='v0.1.14-beta',
    packages=['dismal'],
    install_requires=[
        "numpy",
        "pandas",
        "setuptools",
        "pyranges",
        "scikit-allel",
        "demes",
        "matplotlib",
        "msprime",
        "seaborn",
        "tqdm",
        "tskit"
    ]
)
