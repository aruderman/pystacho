#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pymatgen as mp
import matplotlib.pyplot as plt

from matminer.featurizers.structure import JarvisCFID
import numpy as np
import seaborn as sns

import pymatgen.io.cif as mpcif
from os.path import join
from molmass import Formula
import re

from sklearn.cluster import KMeans,MeanShift
from sklearn import decomposition


def import_dataset(nombre):
    """
    función para importar datasets del materials project o features jarvis
    """
    path = "./datasets/"
    
    if nombre == "MP_db":
    
        mp_files = [pd.read_csv(path + f"mp{s}.csv.bz2") for s in range(1,4)]
        dataset = pd.concat(mp_files, ignore_index=True)
    
    elif nombre == "jarvis":

        jarvis_files = [pd.read_csv(path + f"jarvis{s}.csv.bz2") for s in range(11)]
        dataset = pd.concat(jarvis_files, ignore_index=True)

        jarviscfid = JarvisCFID()
        names = jarviscfid.feature_labels()
        dataset.columns = ['Formula'] + names + ['Energy']

    return dataset        


def get_columns(dataset):
    """
    función que dice el nombre de las columnas del dataset que le paso
    """
    columns = dataset.columns.tolist()
    columns_df = pd.DataFrame({'Columns': columns})
    
    return columns_df


def displot(dataset, column):
    """
    función para graficar la distribución de los valores según la columna que se
    especifique
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
