from setuptools import setup

setup(
    name='dismal',
    version='v0.1.2-alpha',
    packages=['dismal'],
    requires=[
        "numpy",
        "scipy",
        "iclik",
        "prettytable"
    ]
)
