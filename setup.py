from setuptools import setup, find_packages

setup(
    name="model-registry",
    version="0.1.0",
    packages=find_packages(),
    extras_require={"dev": ["pytest"]}
)