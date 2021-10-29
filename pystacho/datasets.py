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

"""The datasets module includes utilities to fetch datasets.

These are from materials project and its projection using the jarvisCFID
from matminer.
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
    """To cache the dataset.

    Parameters
    ----------
    dataset_files : list
        With the file names

    tag : str
        The key of the cache, different for each function that calls
        _from_cache

    cache_path : str (default="~/.pystacho_cache/")
        The path to the directory where the dataset is going to be cached

    force : bool (default=False)
        Force to save the dataset in the cache regardless if the key
        already exists.

    expire : float (default=2.628e6)
        Time (in seconds) to expire the dataset associtated with the key `tag`.


    Returns
    -------
    value : pd.DataFrame
        The result of calling this function or the cached dataframe
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
        print(f"Caching data from: {URL}")
        dataset = [
            pd.read_csv(URL + dfile, compression="bz2")
            for dfile in dataset_files
        ]
        value = pd.concat(dataset, ignore_index=True)

    cache.set(key, value, tag="Dataframe", expire=expire)

    return value


def fetch_mpdb(key="mpdb", **kwargs):
    """Materials Project Database.

    This dataset contains 140000 structures and its calculated properties
    from DFT computations.

    Parameters
    ----------
    key : str (default="mpdb")
        The key of the cache.

    Returns
    -------
    _from_cache : pd.DataFrame
        The result of calling _from_cache function or the cached dataframe
    """
    mp_files = [f"mp{i}.csv.bz2" for i in range(1, 4)]

    return _from_cache(mp_files, key, **kwargs)


def fetch_jarvis(key="jarvis", **kwargs):
    """Jarvis calculations from cif structures of mpdb.

    This dataset contains 42000 crystal structures obteined from the Materials
    Project database and projected into 1555 features using the JarvisCFID()
    featurizer from the matminer library.

    Parameters
    ----------
    key : str (default="jarvis")
        The key of the cache.

    Returns
    -------
    _from_cache : pd.DataFrame
        The result of calling _from_cache function or the cached dataframe
    """
    jarvis_files = [f"jarvis{i}.csv.bz2" for i in range(11)]

    dataset = _from_cache(jarvis_files, key, **kwargs)
    dataset = dataset.drop(dataset.columns[-1], axis=1)
    dataset.columns = ["Formula"] + JarvisCFID().feature_labels()

    return dataset


def fetch_mpdb_filter(key="mp_filter", **kwargs):
    """Materials Project Database filtered.

    This dataset is contains 42000 structures with the same features as the
    original Materials Project dataset. The structures were filtered first
    by e_above_hull < 0.001 eV and then choosing those in which JarvisCFID()
    worked.

    Parameters
    ----------
    key : str (default="mp_filter")
        The key of the cache.

    Returns
    -------
    _from_cache : pd.DataFrame
        The result of calling _from_cache function or the cached dataframe
    """
    filter_file = ["mp_filter.csv.bz2"]

    return _from_cache(filter_file, key, **kwargs)


def fetch_target(target, key=None, **kwargs):
    """Choose an specific target of mpdb.

    It should be determinated by some Machine Learning model.

    Parameters
    ----------
    target : str
        The name of the specific column.

    key : str (default=str(target))
        The key of the cache.

    Returns
    -------
    _from_cache : pd.DataFrame
        The result of calling _from_cache function or the cached dataframe
    """
    target_file = [f"{target}.csv.bz2"]
    tag = f"{target}" if key is None else key

    return _from_cache(target_file, tag, **kwargs)
