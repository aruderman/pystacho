# -*- coding: utf-8 -*-
#
# la primera vez que se corre es muy lento, después queda cacheado
# quizás haya una forma mejor de hacerlo ...

import numpy as np
import pandas as pd

import pystacho.datasets


def test_load_mpdb():

    result = pystacho.datasets.load_mpdb()

    assert isinstance(result, pd.DataFrame)
    np.testing.assert_almost_equal(result["energy"].mean(), -180.367320, 6)
    np.testing.assert_almost_equal(
        result["energy_per_atom"].mean(), -5.903126, 6
    )
    np.testing.assert_almost_equal(
        result["formation_energy_per_atom"].mean(), -1.374002, 6
    )
    assert (
        result["e_above_hull"] >= np.zeros(len(result["e_above_hull"]))
    ).any()


def test_load_mpdb_filter():

    result = pystacho.datasets.load_mpdb_filter()

    assert isinstance(result, pd.DataFrame)
    np.testing.assert_almost_equal(result["energy"].mean(), -165.758806, 6)
    np.testing.assert_almost_equal(
        result["energy_per_atom"].mean(), -5.795536, 6
    )
    np.testing.assert_almost_equal(
        result["formation_energy_per_atom"].mean(), -1.420426, 6
    )
    assert (
        result["e_above_hull"] >= np.zeros(len(result["e_above_hull"]))
    ).any()
