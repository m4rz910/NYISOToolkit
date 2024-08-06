import setuptools
import pathlib as pl

README = pl.Path(".", "README.md").read_text()

setuptools.setup(
    name="nyisotoolkit",
    version="2024.1.0",
    description="A collection of modules for accessing power system data, generating statistics, and creating visualizations from the New York Independent System Operator (NYISO).",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/m4rz910/NYISOToolkit",
    author="m4rz910",
    author_email="viosimosllc@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    package_data = {
        '': ['*.yml']
    },
    install_requires=['pandas==2.2.2',
                      'matplotlib==3.9.1',
                      'scipy==1.14.0',
                      'seaborn==0.13.2',
                      'pytest==8.2.2',
                      'pytz==2024.1',
                      'requests==2.32.0',
                      'pyyaml==6.0.1',
                      'ipykernel',
                      ]
)
