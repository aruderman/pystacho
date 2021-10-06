# import requests
import diskcache as dcache
import pandas as pd
from matminer.featurizers.structure import JarvisCFID

cache = dcache.Cache(directory="./_cache")


def get_mpdb(force=False, expire=6e9):
    url_main = (
        "https://github.com/aruderman/pystacho/raw/main/pystacho/datasets/"
    )
    key = dcache.core.args_to_key(
        base=("pystacho", "mpdb"), args=(url_main,), kwargs={}, typed=False
    )

    cache.expire()
    value = (
        dcache.core.ENOVAL
        if force
        else cache.get(key, default=dcache.core.ENOVAL)
    )

    if value is dcache.core.ENOVAL:
        # response = requests.get(url)
        # value = response.text
        mp_files = []
        for s in range(1, 4):
            filepath = f"mp{s}.csv.bz2"
            print("Caching data:", url_main + filepath)
            mp_files.append(
                pd.read_csv(url_main + filepath, compression="bz2")
            )

        value = pd.concat(mp_files, ignore_index=True)

    cache.set(key, value, tag="Dataframe", expire=expire)

    return value


def get_jarvis(force=False, expire=6e9):
    url_main = (
        "https://github.com/aruderman/pystacho/raw/main/pystacho/datasets/"
    )
    key = dcache.core.args_to_key(
        base=("pystacho", "mpdb"), args=(url_main,), kwargs={}, typed=False
    )

    cache.expire()
    value = (
        dcache.core.ENOVAL
        if force
        else cache.get(key, default=dcache.core.ENOVAL)
    )

    if value is dcache.core.ENOVAL:
        # response = requests.get(url)
        # value = response.text
        jarvis_files = []
        for s in range(11):
            filepath = f"jarvis{s}.csv.bz2"
            print("Caching data:", url_main + filepath)
            jarvis_files.append(
                pd.read_csv(url_main + filepath, compression="bz2")
            )

        dataset = pd.concat(jarvis_files, ignore_index=True)

        jarviscfid = JarvisCFID()

        names = jarviscfid.feature_labels()

        dataset = dataset.drop(dataset.columns[-1], axis=1)
        dataset.columns = ["Formula"] + names

        value = dataset

    cache.set(key, value, tag="Dataframe", expire=expire)

    return value
