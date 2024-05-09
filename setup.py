from setuptools import setup

setup(
    name='dismal',
    version='v0.1.19-beta',
    packages=['dismal'],
    install_requires=[
        "numpy",
        "pandas",
        "setuptools",
        "pyranges==0.0.129",  # bug in 0.1.0
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
