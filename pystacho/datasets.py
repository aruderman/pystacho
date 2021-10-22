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
The datasets module includes utilities to fetch datasets from materials project
and its projection using the jarvisCFID.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

import diskcache as dcache
import pandas as pd
from matminer.featurizers.structure import JarvisCFID

# ============================================================================
# CONSTANTS
# ============================================================================

URL = "https://github.com/aruderman/pystacho_datasets/raw/main/"

PYSTACHO_CACHE_PATH = pathlib.Path(
    os.path.expanduser(os.path.join("~", ".pystacho_cache"))
)

# ============================================================================
# FUNCTIONS
# ============================================================================


def _from_cache(
    dataset_files,
    tag,
    cache_path=PYSTACHO_CACHE_PATH,
    force=False,
    expire=2.628e6,
):
    """
    dataset_files is a list with the file names and tag is the key of the
    cache
    """
    cache = dcache.Cache(directory=cache_path)

    key = dcache.core.args_to_key(
        base=("pystacho", tag), args=(URL,), kwargs={}, typed=False
    )

    cache.expire()
    value = (
        dcache.core.ENOVAL
        if force
        else cache.get(key, default=dcache.core.ENOVAL)
    )

    if value is dcache.core.ENOVAL:
        dataset = []
        for dfile in dataset_files:
            print("Caching data:", URL + dfile)
            dataset.append(pd.read_csv(URL + dfile, compression="bz2"))

        value = pd.concat(dataset, ignore_index=True)

    cache.set(key, value, tag="Dataframe", expire=expire)

    return value


def fetch_mpdb(key="mpdb", **kwargs):
    """
    This dataset contains 140000 Materials Project structures and its
    calculated properties.
    """
    mp_files = [f"mp{i}.csv.bz2" for i in range(1, 4)]

    return _from_cache(mp_files, key, **kwargs)


def fetch_jarvis(key="jarvis", **kwargs):
    """
    This dataset contains 42000 crystal structures obteined from the Materials
    Project database and projected into 1555 features using the JarvisCFID()
    featurizer from the matminer library
    """
    jarvis_files = [f"jarvis{i}.csv.bz2" for i in range(11)]

    dataset = _from_cache(jarvis_files, key, **kwargs)
    dataset = dataset.drop(dataset.columns[-1], axis=1)
    dataset.columns = ["Formula"] + JarvisCFID().feature_labels()

    return dataset


def fetch_mpdb_filter(key="mp_filter", **kwargs):
    """
    This dataset is contains 42000 structures with the same features as the
    original Materials Project dataset.
    The structures were filtered first by e_above_hull < 0.001 eV and then
    choosing those in which JarvisCFID()
    worked
    """
    filter_file = ["mp_filter.csv.bz2"]

    return _from_cache(filter_file, key, **kwargs)


def fetch_target(target, key=None, **kwargs):
    """
    Load the Materials Project dataset column chosen as target for ML
    """
    target_file = [f"{target}.csv.bz2"]
    tag = f"{target}" if key is None else key

    return _from_cache(target_file, tag, **kwargs)
