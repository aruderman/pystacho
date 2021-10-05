import requests
import diskcache as dcache
import pandas as pd

cache = dcache.Cache(directory="./_cache")

def get(url, force=False):

    key = dcache.core.args_to_key(
        base=("pystacho", "ejemplo"), args=(url,), kwargs={}, typed=False
    )

    cache.expire()
    value = dcache.core.ENOVAL if force else cache.get(key, default=dcache.core.ENOVAL)

    if value is dcache.core.ENOVAL:
        #response = requests.get(url)
        #value = response.text
        print("Caching data:", url)
        df = pd.read_csv(url)
        value = df
        cache.set(key, value, tag="Dataframe", expire=5)

    return value


#Main
url = "https://raw.githubusercontent.com/marcolongus/data/main/elasticity.elastic_anisotropy.csv"
print(type(get(url)))

