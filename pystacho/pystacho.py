#!/usr/bin/env python
# coding: utf-8
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatgen as mp
import seaborn as sns
from lightgbm.sklearn import LGBMRegressor
from matminer.featurizers.structure import JarvisCFID
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer


def import_dataset(nombre):
    """
    función para importar datasets del materials project o features jarvis
    """
    path = os.path.abspath(os.path.dirname(__file__)) + "/datasets/"
    if nombre == "MP_db":

        mp_files = [pd.read_csv(path + f"mp{s}.csv.bz2") for s in range(1, 4)]
        dataset = pd.concat(mp_files, ignore_index=True)

    elif nombre == "MP_filter":

        dataset = pd.read_csv(path + "mp_filter.csv.bz2", ignore_index=True)

    elif nombre == "jarvis":

        jarvis_files = [
            pd.read_csv(path + f"jarvis{s}.csv.bz2") for s in range(11)
        ]
        dataset = pd.concat(jarvis_files, ignore_index=True)

        jarviscfid = JarvisCFID()
        names = jarviscfid.feature_labels()
        dataset = dataset.drop(dataset.columns[-1], axis=1)
        dataset.columns = ["Formula"] + names

    return dataset


def load_target(target):
    """ "
    Cargo las variables de target para hacer las predicciones y calcular las
    features relevantes por target
    """
    target = pd.read_csv(f"./target/{target}.csv")
    return target


def get_important_features(model, target, n_jobs, n_features):
    """
    Selecciono las filas de jarvis con valores en la variable target y
    calculo las features relevantes para esa variable
    """
    target = load_target(target)
    y = target.iloc[:, -1].tolist()

    jarvis = import_dataset("jarvis")
    x = jarvis[~target.iloc[:, -1].isnull()]
    x = x.drop(x.columns[0], axis=1)

    standard = Normalizer()
    x = pd.DataFrame(standard.fit_transform(x))

    jarviscfid = JarvisCFID()
    names = jarviscfid.feature_labels()
    x.columns = names

    if model == "lgbm":
        fit_model = LGBMRegressor(n_estimators=2000, n_jobs=n_jobs)
        fit_model.fit(x, y)

    best_features_index = np.absolute(fit_model.feature_importances_).argsort()
    best_features_index = best_features_index[-n_features:][::1]
    best_features_values = fit_model.feature_importances_[best_features_index]
    best_features_names = x.iloc[:, best_features_index].columns

    return best_features_names, best_features_values


def plot_best_features(best_features_names, best_features_values):
    plt.figure(figsize=(10, 6))
    sns.set(font_scale=1.2)
    sns.barplot(y=best_features_names, x=best_features_values)
    plt.show()


def train_model(model, target, best_features_names, n_jobs, **kwargs):
    """
    función que entrena los modelos según los targets y las best_features
    """
    target = load_target(target)
    y = target.iloc[:, -1].tolist()

    jarvis = import_dataset("jarvis")
    x = jarvis[~target.iloc[:, -1].isnull()]
    x = x.drop(x.columns[0], axis=1)

    standard = Normalizer()
    x = pd.DataFrame(standard.fit_transform(x))

    jarviscfid = JarvisCFID()
    names = jarviscfid.feature_labels()
    x.columns = names

    x = x[[best_features_names]]
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=0
    )

    if model == "lgbm":
        # acá faltaría agregar más parámetros
        fit_model = LGBMRegressor(n_estimators=2000, n_jobs=n_jobs)
        fit_model.fit(x_train, y_train)
        y_train_pred = fit_model.predict(x_train)
        y_valid_pred = fit_model.predict(x_valid)

        print("Conjunto de entrenamiento: modelo LGBMRegressor_red")
        print("R2: ", r2_score(y_train, y_train_pred))
        print("MAE: ", mean_absolute_error(y_train, y_train_pred))
        print(
            "MSE: ", mean_squared_error(y_train, y_train_pred, squared=False)
        )

        print("Conjunto de validación: modelo LGBMRegressor_red")
        print("R2: ", r2_score(y_valid, y_valid_pred))
        print("MAE: ", mean_absolute_error(y_valid, y_valid_pred))
        print(
            "MSE: ", mean_squared_error(y_valid, y_valid_pred, squared=False)
        )

    return fit_model, standard


def from_cif_to_jarvis(path, cif):
    """
    transforma un archivo cif en una fila jarvis
    """
    jarvis_features = []

    jarviscfid = JarvisCFID()
    # names no se usa ??
    names = jarviscfid.feature_labels()

    cif_structure = mp.Structure.from_file(os.join(path, cif))
    cif_feature = jarviscfid.featurize(cif_structure)

    jarvis_features.append(cif_feature)
    jarvis_features.columns = names

    return jarvis_features


def fit_data(path, cif, fit_model, best_features_names, standard):
    """
    función que predice el valor de la energía para un dado cif
    """
    jarvis_features = from_cif_to_jarvis(path, cif)
    jarvis_features = jarvis_features[[best_features_names]]

    standard = Normalizer()
    jarvis_features = pd.DataFrame(standard.transform(jarvis_features))

    predict = fit_model.predict(jarvis_features)

    return predict


def get_columns(dataset):
    """
    función que dice el nombre de las columnas del dataset que le paso
    """
    columns = dataset.columns.tolist()
    columns_df = pd.DataFrame({"Columns": columns})

    return columns_df


def displot(dataset, column):
    """
    función para graficar la distribución de los valores según la columna que
    se especifique
    """
    ax = sns.displot(data=dataset, x=column)
    return ax


def displot2d(dataset, column1, column2, kind):
    """
    función para gráficar una distribución de valores 2D a partir de las dos
    columnas especificadas
    """
    ax = sns.displot(data=dataset, x=column1, y=column2, kind=kind)
    return ax


def cluster_inertia(dataset, column1, column2):
    """
    cluster inertia KMeans
    """
    dataset = dataset[[column1, column2]]
    scores = [
        KMeans(n_clusters=i + 2).fit(dataset).inertia_ for i in range(10)
    ]
    plt.plot(np.arange(2, 12), scores)
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Inertia of k-Means versus number of clusters")
    plt.show()


def plot_clusters(dataset, column1, column2, n_clusters):
    """
    plot KMeans
    """
    dataset = dataset[[column1, column2]]
    labels = KMeans(n_clusters).fit(dataset).labels_
    ax = sns.scatterplot(data=dataset, x=column1, y=column2, hue=labels)
    return ax


if __name__ == "__main__":

    # main de materials project
    MP_db = import_dataset(nombre="MP_db")
    print(MP_db.head())
    print(get_columns(MP_db))
    ax = displot(MP_db, "energy_per_atom")
    plt.show()
    ax = displot2d(MP_db, "energy", "energy_per_atom", "kde")
    plt.show()
    cluster_inertia(MP_db, "energy", "energy_per_atom")
    plot_clusters(MP_db, "energy", "energy_per_atom", 5)
    plt.show()

    # main jarvis
    jarvis = import_dataset(nombre="jarvis")
    print(jarvis)