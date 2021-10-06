#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Pystacho Project (https://github.com/aruderman/pystacho/).
# Copyright (c) 2021, Francisco Fernandez, Benjamin Marcologno, Andrés Ruderman
# License: MIT
#   Full Text: https://github.com/aruderman/pystacho/blob/master/LICENSE

# =====================================================================
# DOCS
# =====================================================================

"""This file is for distribute and install Pystacho"""

# ======================================================================
# IMPORTS
# ======================================================================

import os
import pathlib

from setuptools import setup

# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


REQUIREMENTS = [
    "diskcache",
    "numpy",
    "pandas",
    "matplotlib",
    "pymatgen",
    "seaborn",
    "lightgbm",
    "matminer",
    "scikit-learn",
]

with open(PATH / "pystacho" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break


with open("README.md") as fp:
    LONG_DESCRIPTION = fp.read()


# =============================================================================
# FUNCTIONS
# =============================================================================

setup(
    name="Pystacho",
    version=VERSION,
    description="ESCRIBIR DESCRIPCION DEL PROYECTO",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=["Francisco Fernandez", "Benjamin Marcologno", "Andrés Ruderman"],
    author_email="andres.ruderman@gmail.com",
    url="https://github.com/aruderman/pystacho",
    packages=["pystacho"],
    license="The MIT License",
    install_requires=REQUIREMENTS,
    keywords=["pystacho"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
    ],
    # include_package_data=True,
)
