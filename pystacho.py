#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
from matminer.featurizers.structure import JarvisCFID
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import train_test_split


def import_dataset(nombre):
    """
    función para importar datasets del materials project o features jarvis
    """
    path = "./datasets/"
    if nombre == "MP_db":

        mp_files = [pd.read_csv(path + f"mp{s}.csv.bz2") for s in range(1, 4)]
        dataset = pd.concat(mp_files, ignore_index=True)

    elif nombre == 'MP_filter':

        dataset = pd.read_csv(path + "mp_filter.csv.bz2", ignore_index=True)

    elif nombre == "jarvis":

        jarvis_files = [
                pd.read_csv(path + f"jarvis{s}.csv.bz2") for s in range(11)
        ]
        dataset = pd.concat(jarvis_files, ignore_index=True)

        jarviscfid = JarvisCFID()
        names = jarviscfid.feature_labels()
        dataset = dataset.drop(dataset.columns[-1], axis=1)
        dataset.columns = ['Formula'] + names

    return dataset


def load_target(target):
    """"
    Cargo las variables de target para hacer las predicciones y calcular las
    features relevantes por target
    """
    target = pd.read_csv(f'./target/{target}.csv')
    return target


def get_important_features(model, target, n_jobs, n_features):
    """
    Selecciono las filas de jarvis con valores en la variable target y
    calculo las features relevantes para esa variable
    """
    y = target.iloc[:, -1].tolist()
    x = jarvis[~target.iloc[:, -1].isnull()]
    # ERROR! dataset no está definido y tampoco se lo está pasando como
    # argumento de la función, esto tira error
    x = x.drop(dataset.columns[0], axis=1)

    standard = Normalizer()
    x = pd.DataFrame(standard.fit_transform(x))
    # ERROR! jarviscfid no está definida en esta función y tampoco se la
    # pasa como argumento
    names = jarviscfid.feature_labels()

    X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2,
                                                          random_state=0)
    lgbm = LGBMRegressor(n_estimators=2000, n_jobs=n_jobs)
    lgbm.fit(X_train, y_train)
    x.columns = names

    best_features_index = np.absolute(lgbm.feature_importances_).argsort()
    best_features_index = best_features_index[-n_features:][::1]
    best_features_values = lgbm.feature_importances_[best_features_index]
    best_features_names = x.iloc[:, best_features_index].columns

    return best_features_names, best_features_values


def plot_best_features(best_features_names, best_features_values):
    plt.figure(figsize=(10, 6))
    sns.set(font_scale=1.2)
    sns.barplot(y=best_features_names, x=best_features_values)
    plt.show()


def get_columns(dataset):
    """
    función que dice el nombre de las columnas del dataset que le paso
    """
    columns = dataset.columns.tolist()
    columns_df = pd.DataFrame({'Columns': columns})

    return columns_df


def displot(dataset, column):
    """
    función para graficar la distribución de los valores según la columna que
    se especifique
    """
    ax = sns.displot(data=dataset, x=column)
    plt.show()


def displot2D(dataset, column1, column2, kind):
    """
    función para gráficar una distribución de valores 2D a partir de las dos
    columnas especificadas
    """
    ax = sns.displot(data=dataset, x=column1, y=column2, kind=kind)
    plt.show()


def cluster_inertia(dataset, column1, column2):
    """
    cluster inertia KMeans
    """
    dataset = dataset[[column1, column2]]
    scores = [KMeans(n_clusters=i+2).fit(dataset).inertia_ for i in range(10)]
    plt.plot(np.arange(2, 12), scores)
    plt.xlabel('Number of clusters')
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
    plt.show()


if __name__ == "__main__":

    # main de materials project
    MP_db = import_dataset(nombre='MP_db')
    print(MP_db.head())
    print(get_columns(MP_db))
    displot(MP_db, 'energy_per_atom')
    displot2D(MP_db, 'energy', 'energy_per_atom', 'kde')
    cluster_inertia(MP_db, 'energy', 'energy_per_atom')
    plot_clusters(MP_db, 'energy', 'energy_per_atom', 5)

    # main jarvis
    jarvis = import_dataset(nombre='jarvis')
    print(jarvis)
