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
The datasets module includes utilities to load datasets from materials project
and its projection using the jarvisCFID.
"""

# ============================================================================
# IMPORTS
# ============================================================================

# import requests
import diskcache as dcache
import pandas as pd
from matminer.featurizers.structure import JarvisCFID

# ============================================================================
# CONSTANTS
# ============================================================================

CACHE = dcache.Cache(directory="./_cache")
URL = "https://github.com/aruderman/pystacho_datasets/raw/main/"

# ============================================================================
# FUNCTIONS
# ============================================================================


def get(dataset_files, key="example", force=False, expire=2.628e6):
    """
    dataset_files is a list with the file names
    """
    key = dcache.core.args_to_key(
        base=("pystacho", key), args=(URL,), kwargs={}, typed=False
    )

    CACHE.expire()
    value = (
        dcache.core.ENOVAL
        if force
        else CACHE.get(key, default=dcache.core.ENOVAL)
    )

    if value is dcache.core.ENOVAL:
        # response = requests.get(url)
        # value = response.text
        dataset = []
        for dfile in dataset_files:
            print("Caching data:", URL + dfile)
            dataset.append(pd.read_csv(URL + dfile, compression="bz2"))

        value = pd.concat(dataset, ignore_index=True)

    CACHE.set(key, value, tag="Dataframe", expire=expire)

    return value


def load_mpdb(key="mpdb", **kwargs):
    """
    This dataset contains 140000 Materials Project structures and its
    calculated properties.
    """
    mp_files = [f"mp{i}.csv.bz2" for i in range(1, 4)]

    return get(mp_files, key=key, **kwargs)


def load_jarvis(key="jarvis", **kwargs):
    """
    This dataset contains 42000 crystal structures obteined from the Materials
    Project database and projected into 1555 features using the JarvisCFID()
    featurizer from the matminer library
    """
    jarvis_files = [f"jarvis{i}.csv.bz2" for i in range(11)]

    dataset = get(jarvis_files, key=key, **kwargs)

    jarviscfid = JarvisCFID()

    names = jarviscfid.feature_labels()

    dataset = dataset.drop(dataset.columns[-1], axis=1)
    dataset.columns = ["Formula"] + names

    return dataset


def load_mpdb_filter(key="mp_filter", **kwargs):
    """
    This dataset is contains 42000 structures with the same features as the
    original Materials Project dataset.
    The structures were filtered first by e_above_hull < 0.001 eV and then
    choosing those in which JarvisCFID()
    worked
    """
    filter_file = ["mp_filter.csv.bz2"]

    return get(filter_file, key=key, **kwargs)


def load_target(target, key=None, **kwargs):
    """
    Load the Materials Project dataset column chosen as target for ML
    """
    target_file = [f"{target}.csv"]
    if key is None:
        key = f"{target}"

    return get(target_file, key=key, **kwargs)
