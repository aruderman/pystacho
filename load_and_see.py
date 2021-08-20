#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pymatgen as mp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matminer.featurizers.structure import JarvisCFID
import numpy as np
import seaborn as sns

import pymatgen.io.cif as mpcif
from os.path import join
from molmass import Formula
import re

from sklearn.cluster import KMeans,MeanShift
from sklearn import decomposition



# importo el dataset dematerials project
def import_dataset(nombre):
    if nombre == 'MP_db':
        path_MP_db = './MP-db/'
        dataset = pd.concat([pd.read_csv(path_MP_db+'dataset1.csv'), pd.read_csv(path_MP_db+'dataset2.csv'), pd.read_csv(path_MP_db+'dataset3.csv')], ignore_index=True)
    elif nombre == 'jarvis':
        path_jarvis = './jarvis_features/'
        dataset = pd.concat([pd.read_csv(path_jarvis+'jarvis1.csv'), pd.read_csv(path_jarvis+'jarvis2.csv'), pd.read_csv(path_jarvis+'jarvis3.csv'), 
                             pd.read_csv(path_jarvis+'jarvis4.csv')], ignore_index=True)
        jarviscfid = JarvisCFID()
        names = jarviscfid.feature_labels()
        dataset.columns = ['Formula'] + names + ['Energy']
    return dataset        



MP_db = import_dataset(nombre='MP_db')
MP_db.head()


jarvis = import_dataset(nombre='jarvis')
jarvis



#función que me dice el nombre de las columnas del data set

def get_columns(dataset):
    columns = dataset.columns.tolist()
    columns_df = pd.DataFrame({'Columns': columns})
    return columns_df

get_columns(MP_db)


#Gráfico de distribución de valores

def displot(dataset, column):
    ax = sns.displot(data=dataset, x=column)
    plt.show()

displot(MP_db, 'energy_per_atom')


#Gráfico de distribución de valores 2D

def displot2D(dataset, column1, column2, kind):
    ax = sns.displot(data=dataset, x=column1, y=column2, kind=kind)
    plt.show()


displot2D(MP_db, 'energy', 'energy_per_atom', 'kde')


def cluster_inertia(dataset, column1, column2):
    dataset = dataset[[column1, column2]]
    scores = [KMeans(n_clusters=i+2).fit(dataset).inertia_ for i in range(10)]
    plt.plot(np.arange(2, 12), scores)
    plt.xlabel('Number of clusters')
    plt.ylabel("Inertia")
    plt.title("Inertia of k-Means versus number of clusters")
    plt.show()


cluster_inertia(MP_db, 'energy', 'energy_per_atom')


def plot_clusters(dataset, column1, column2, n_clusters):
    dataset = dataset[[column1, column2]]
    labels = KMeans(n_clusters).fit(dataset).labels_
    ax = sns.scatterplot(data=dataset, x=column1, y=column2, hue=labels)
    plt.show()


plot_clusters(MP_db, 'energy', 'energy_per_atom', 5)

