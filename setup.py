# -*- coding: utf-8 -*-

import setuptools

setuptools.setup(
    name="local_nyisotoolkit",
    version="1.0.0",
    author="m4rz910",
    author_email="viosimosllc@gmail.com",
    description="A collection of modules for accessing power system data, generating statistics, and creating visualizations from the New York Independent System Operator (NYISO).",
    packages=setuptools.find_packages(),
    # py_modules=['nyisotoolkit'],
    url="https://github.com/m4rz910/NYISOToolkit",    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)