#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   pystacho Project (https://github.com/aruderman/pystacho/).
# Copyright (c) 2021, Francisco Fernandez, Benjamín Marcolongo, Andrés Ruderman
# License: MIT
#   Full Text: https://github.com/aruderman/pystacho/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""
The pystacho.datasets module includes utilities to load datasets from materials
project and its projection using the jarvisCFID.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

import pandas as pd

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# ============================================================================
# FUNCTIONS
# ============================================================================


def load_jarvis(nombre):
    """
    This dataset contains 42000 crystal structures obteined from the Materials
    Project database and projected into 1555 features using the JarvisCFID()
    featurizer from the matminer library
    """
    jarvis_files = [
        pd.read_csv(PATH + f"jarvis{s}.csv.bz2") for s in range(11)
    ]
    dataset = pd.concat(jarvis_files, ignore_index=True)

    from matminer.featurizers.structure import JarvisCFID

    jarviscfid = JarvisCFID()

    names = jarviscfid.feature_labels()

    dataset = dataset.drop(dataset.columns[-1], axis=1)
    dataset.columns = ["Formula"] + names

    return dataset


def load_mpdb(nombre):
    """
    This dataset contains 140000 Materials Project structures and its
    calculated properties.
    """
    mp_files = [
        pd.read_csv(PATH + f"mp{s}.csv.bz2", compression="bz2")
        for s in range(1, 4)
    ]
    dataset = pd.concat(mp_files, ignore_index=True)

    return dataset


def load_mpdb_filter(nombre):
    """
    This dataset is contains 42000 structures with the same features as the
    original Materials Project dataset.
    The structures were filtered first by e_above_hull < 0.001 eV and then
    choosing those in which JarvisCFID()
    worked
    """
    dataset = pd.read_csv(
        PATH + "mp_filter.csv.bz2", ignore_index=True, compression="bz2"
    )

    return dataset


def load_target(target):
    """ "
    Load the Materials Project dataset column chosen as target for ML
    """
    target = pd.read_csv(f"./target/{target}.csv")
    return target
