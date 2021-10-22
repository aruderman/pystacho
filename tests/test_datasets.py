# -*- coding: utf-8 -*-
#
# HAY QUE ROBUSTECER ESTOS TESTS
#
# la primera vez que se corre es muy lento, después queda cacheado
# quizás haya una forma mejor de hacerlo ...

import os
import pathlib
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from matminer.featurizers.structure import JarvisCFID

import pystacho.datasets

TEST_DATA = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
)


def test_fetch_mpdb():

    mp_path = TEST_DATA / "mp_test.csv.bz2"

    mock_df = pd.read_csv(mp_path, compression="bz2")

    with mock.patch("pandas.read_csv", return_value=mock_df):
        result = pystacho.datasets.fetch_mpdb(force=True)

    assert isinstance(result, pd.DataFrame)
    np.testing.assert_almost_equal(result["energy"].mean(), -15.616882, 6)
    np.testing.assert_almost_equal(
        result["energy_per_atom"].mean(), -5.707032, 6
    )
    np.testing.assert_almost_equal(
        result["formation_energy_per_atom"].mean(), -0.539429, 6
    )
    np.testing.assert_almost_equal(result["e_above_hull"].mean(), 0.068912, 6)


def test_fetch_jarvis():

    jarvis_path = TEST_DATA / "jarvis_test.csv.bz2"
    jarvis_labels = ["Formula"] + JarvisCFID().feature_labels()

    # saco la columna de los indices de mock_df porque en la base de jarvis
    # no está, pero en jarvis_test sí...
    mock_df = pd.read_csv(jarvis_path, compression="bz2")
    mock_df = mock_df.drop(mock_df.columns[0], axis=1)

    with mock.patch("pandas.read_csv", return_value=mock_df):
        result = pystacho.datasets.fetch_jarvis(force=True)

    assert isinstance(result, pd.DataFrame)
    assert (result.keys() == jarvis_labels).all()
    np.testing.assert_almost_equal(
        result["jml_bp_mult_atom_rad"].mean(), 3196.4154, 4
    )
    np.testing.assert_almost_equal(
        result["jml_hfus_add_bp"].mean(), 2041.962881, 6
    )
    np.testing.assert_almost_equal(
        result["jml_elec_aff_mult_voro_coord"].mean(), 4.2255, 4
    )
    np.testing.assert_almost_equal(
        result["jml_mol_vol_subs_atom_mass"].mean(), -75.310246, 6
    )


def test_fetch_mpdb_filter():

    mp_filter_path = TEST_DATA / "mp_filter_test.csv.bz2"

    mock_df = pd.read_csv(mp_filter_path, compression="bz2")

    with mock.patch("pandas.read_csv", return_value=mock_df):
        result = pystacho.datasets.fetch_mpdb_filter(force=True)

    assert isinstance(result, pd.DataFrame)
    np.testing.assert_almost_equal(result["energy"].mean(), -57.634227, 6)
    np.testing.assert_almost_equal(
        result["energy_per_atom"].mean(), -6.673063, 6
    )
    np.testing.assert_almost_equal(
        result["formation_energy_per_atom"].mean(), -0.904769, 6
    )
    assert (
        result["e_above_hull"] >= np.zeros(len(result["e_above_hull"]))
    ).all()


@pytest.mark.parametrize(
    "target_name, mean_value",
    [
        ("elasticity.elastic_anisotropy", 4.866),
        ("elasticity.G_Reuss", 54.666667),
        ("elasticity.G_Voigt", 67.766667),
        ("elasticity.G_Voigt_Reuss_Hill", 61.133333),
        ("elasticity.G_VRH", 61.133333),
        ("elasticity.K_Reuss", 95.533333),
        ("elasticity.K_Voigt", 107.033333),
        ("elasticity.K_Voigt_Reuss_Hill", 101.266667),
        ("elasticity.K_VRH", 101.266667),
        ("energy", -89.108629),
        ("energy_per_atom", -6.168534),
        ("formation_energy_per_atom", -1.047940),
    ],
)
def test_fetch_target(target_name, mean_value):

    target_path = TEST_DATA / f"{target_name}_test.csv.bz2"

    mock_df = pd.read_csv(target_path, compression="bz2")

    with mock.patch("pandas.read_csv", return_value=mock_df):
        result = pystacho.datasets.fetch_target(target_name, force=True)

    assert isinstance(result, pd.DataFrame)
    np.testing.assert_almost_equal(result[target_name].mean(), mean_value, 6)
