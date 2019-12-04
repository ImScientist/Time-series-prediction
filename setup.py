import io
import os
from setuptools import find_packages, setup

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


with io.open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name="stargazers",
    description="Poc stargazers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    zip_safe=False,
    install_requires=requirements,
    author="Anton Ivanov",
    author_email="a.i.ivanov.sv@gmail.com",
    classifiers=["Python :: 3.7 :: time series"],
)
