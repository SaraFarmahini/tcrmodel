from setuptools import setup, find_packages

setup(
    name="tcrmodels",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "scikit-learn",
        "tqdm",
        "pytorch_lightning"
    ],
) 