# -*- coding: utf-8 -*-

import pandas as pd

import pystacho


# este test es muy lento para nuestro caso que la base de datos es grande
# como hacemos para mejorar?
def test_load_mpdb():

    result = pystacho.datasets.load_mpdb()

    assert isinstance(result, pd.DataFrame)
