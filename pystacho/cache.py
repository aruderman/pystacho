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
The cache module includes utilities to load datasets from materials project
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


def get_mpdb(force=False, expire=2.628e6):
    """
    This dataset contains 140000 Materials Project structures and its
    calculated properties.
    """
    key = dcache.core.args_to_key(
        base=("pystacho", "mpdb"), args=(URL,), kwargs={}, typed=False
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
        mp_files = []
        for s in range(1, 4):
            filepath = f"mp{s}.csv.bz2"
            print("Caching data:", URL + filepath)
            mp_files.append(pd.read_csv(URL + filepath, compression="bz2"))

        value = pd.concat(mp_files, ignore_index=True)

    CACHE.set(key, value, tag="Dataframe", expire=expire)

    return value


def get_jarvis(force=False, expire=2.628e6):
    """
    This dataset contains 42000 crystal structures obteined from the Materials
    Project database and projected into 1555 features using the JarvisCFID()
    featurizer from the matminer library
    """
    key = dcache.core.args_to_key(
        base=("pystacho", "jarvis"), args=(URL,), kwargs={}, typed=False
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
        jarvis_files = []
        for s in range(11):
            filepath = f"jarvis{s}.csv.bz2"
            print("Caching data:", URL + filepath)
            jarvis_files.append(pd.read_csv(URL + filepath, compression="bz2"))

        dataset = pd.concat(jarvis_files, ignore_index=True)

        jarviscfid = JarvisCFID()

        names = jarviscfid.feature_labels()

        dataset = dataset.drop(dataset.columns[-1], axis=1)
        dataset.columns = ["Formula"] + names

        value = dataset

    CACHE.set(key, value, tag="Dataframe", expire=expire)

    return value


def get_mpdb_filter(force=False, expire=2.628e6):
    """
    This dataset is contains 42000 structures with the same features as the
    original Materials Project dataset.
    The structures were filtered first by e_above_hull < 0.001 eV and then
    choosing those in which JarvisCFID()
    worked
    """
    key = dcache.core.args_to_key(
        base=("pystacho", "mpdb_filter"), args=(URL,), kwargs={}, typed=False
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
        print("Caching data:", URL + "mp_filter.csv.bz2")
        mp_filter = pd.read_csv(URL + "mp_filter.csv.bz2", ignore_index=True, compression="bz2")

        value = mp_filter

    CACHE.set(key, value, tag="Dataframe", expire=expire)

    return value


def get_target(target, force=False, expire=2.628e6):
    """
    Load the Materials Project dataset column chosen as target for ML
    """
    key = dcache.core.args_to_key(
        base=("pystacho", "target"), args=(URL,), kwargs={}, typed=False
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
        print("Caching data:", URL + f"{target}.csv")
        mp_target = pd.read_csv(URL + f"{target}.csv")

        value = mpi_target

    CACHE.set(key, value, tag="Dataframe", expire=expire)

    return value
