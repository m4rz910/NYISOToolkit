# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NYISOToolkit",
    version="1.0.0",
    author="m4rz910",
    author_email="viosimosllc@gmail.com",
    description="A collection of modules for accessing power system data, generating statistics, and creating visualizations from the New York Independent System Operator (NYISO).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m4rz910/NYISOToolkit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)