import setuptools
import pathlib as pl

README = pl.Path('.','README.md').read_text()

setuptools.setup(
    name="nyisotoolkit",
    version="1.0.0",
    
    description="A collection of modules for accessing power system data, generating statistics, and creating visualizations from the New York Independent System Operator (NYISO).",
    long_description=README,
    long_description_content_type = 'text/markdown',
    url="https://github.com/m4rz910/NYISOToolkit",  
    author="m4rz910",
    author_email="viosimosllc@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    packages=setuptools.find_packages(),
    install_requires=['pandas>=1.0.5',
                      'matplotlib>=3.2.2',
                      'pytest>=6.0.1',
                      'pytz>=2020.1',
                      'requests>=2.24.0',
                      'pyyaml>=5.3.1']
)