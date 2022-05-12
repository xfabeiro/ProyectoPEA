# Codigo por Xoaquin Fabeiro Monteagudo

from concurrent.futures import thread
from email import header
from multiprocessing.dummy import Process
from statistics import mode
from threading import Thread
from time import time
from timeit import timeit
from certifi import where
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from requests import head
import scipy.stats as stats
from random import randint
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA, FastICA
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from tensorflow import keras
from keras import layers, models
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from multiprocessing import Process
import warnings
from Proyecto2_XFM import ModeloSeparacionTestTrain, entrenoRN, lecturaeimputaciondatos, modeloRN

# Función para lanzar como hilo de los tres primeros modelos
def find_par1_thread(el, datos, pot, redDim, writer):

    # Búsqueda de mejores parámetros
    # Regresión Logística:
    scoresLR = []
    for i in range(1, 10):
        # Se define el modelo con una normalización y reducción de parámetros previos
        LRaux = make_pipeline(
            StandardScaler(),
            redDim(n_components=el),
            LogisticRegression(penalty="none", solver="lbfgs", max_iter=i * 1000, multi_class="multinomial"),
        )
        # Se llama a la función de entrenamiento del modelo
        scoreLR, scoreMLR = ModeloSeparacionTestTrain(LRaux, "LR", datos, pot, 0.8)
        scoresLR.append(scoreMLR)
    # Se crea un dataframe con los resultados obtenidos
    scoresLR = pd.DataFrame(scoresLR, index=range(1000, 10000, 1000))
    scoresLR.index.name = "Max iter"

    # KNN:
    scoresKNN = []
    for i in range(1, 20):
        # Se define el modelo con una normalización y reducción de parámetros previos
        KNNaux = make_pipeline(StandardScaler(), redDim(n_components=el), KNeighborsClassifier(n_neighbors=1 + i * 2))
        # Se llama a la función de entrenamiento del modelo
        scoreKNN, scoreMKNN = ModeloSeparacionTestTrain(KNNaux, "KNN", datos, pot, 0.8)
        scoresKNN.append(scoreMKNN)
    # Se crea un dataframe con los resultados obtenidos
    scoresKNN = pd.DataFrame(scoresKNN, index=range(3, 2 * 20, 2))
    scoresKNN.index.name = "Neighbors"

    # DT:
    scoresDT = []
    for i in range(1, 20):
        # Se define el modelo con una normalización y reducción de parámetros previos
        DTaux = make_pipeline(
            StandardScaler(), redDim(n_components=el), DecisionTreeClassifier(criterion="entropy", max_depth=1 + i * 2)
        )
        # Se llama a la función de entrenamiento del modelo
        scoreDT, scoreDTM = ModeloSeparacionTestTrain(DTaux, "DT", datos, pot, 0.8)
        scoresDT.append(scoreDTM)
    # Se crea un dataframe con los resultados obtenidos
    scoresDT = pd.DataFrame(scoresDT, index=range(3, 2 * 20, 2))
    scoresDT.index.name = "Max depth"

    # Se escriben en el excel los resultados obtenidos
    scoresLR.to_excel(writer, sheet_name=("LR " + str(redDim)[-5:-2] + "(" + str(el) + ")"))
    scoresKNN.to_excel(writer, sheet_name=("KNN " + str(redDim)[-5:-2] + "(" + str(el) + ")"))
    scoresDT.to_excel(writer, sheet_name=("DT " + str(redDim)[-5:-2] + "(" + str(el) + ")"))


# Función para lanzar como hilo para la Red Neuronal
def find_par2_thread(ncapas, datos, pot, writer):
    # Se aplica una normalización y una reducción de parámetros a los datos y se transforman
    new_datos = make_pipeline(StandardScaler(), PCA(n_components=5)).fit_transform(datos)
    n, m = new_datos.shape
    resDF = []
    for n in range(1, 31):
        neur = []
        # En función del número de capas se crean neuronas aleatorias por capa de entre 3 y 15
        for i in range(ncapas):
            neur.append(randint(3, 15))
        neur.append(2)
        # Se realiza el entreno de la red tres veces, una vez por cada learning rate
        for j in range(1, 4):
            res = [neur, j * 0.001]
            modeloRNaux = modeloRN(neur, j * 0.001, m)
            resDF.append(res + entrenoRN(modeloRNaux, new_datos, pot, 0.8))
    # Se crea el Dataframe para escribirlo en el excel con los resultados
    resDF = pd.DataFrame(
        resDF, columns=["NeurxCapa", "Learning Rate", "Loss", "Accuracy", "Recall", "Precision", "AUC"]
    )
    # Se escriben los resultados en una hoja del documento de excel
    resDF.to_excel(writer, sheet_name=str(ncapas) + " Capas")


# Función para lanzar los diferentes hilos de la primera búsqueda
def find_threads1():
    with pd.ExcelWriter("Busquedaparametros1.xlsx", mode="a", if_sheet_exists="replace") as writer:
        reds = [PCA, FastICA]
        threads1 = []
        for red in reds:
            for el in range(1, 10):
                t1 = Thread(target=find_par1_thread, args=(el, datos, pot, red, writer))
                t1.start()
                threads1.append(t1)
        for t in threads1:
            t.join()


# Función para lanzar la primera búsqueda pero de forma secuencial
def find_sequential1():
    with pd.ExcelWriter("Busquedaparametros1.xlsx", mode="a", if_sheet_exists="replace") as writer:
        reds = [PCA, FastICA]
        for red in reds:
            for el in range(1, 10):
                find_par1_thread(el, datos, pot, red, writer)


# Función para lanzar los diferentes hilos de la segunda búsqueda
def find_threads2():
    with pd.ExcelWriter("Busquedaparametros2.xlsx", mode="a", if_sheet_exists="replace") as writer:
        threads2 = []
        for capas in range(2, 4):
            t2 = Thread(target=find_par2_thread, args=(capas, datos, pot, writer))
            t2.start()
            threads2.append(t2)
        for t in threads2:
            t.join()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Se define el path del conjunto de datos
    datapath1 = os.path.abspath("D:/Universitas/Master_IIR/Master/AA2/Proyecto/waterpotability.csv")
    datapath2 = os.path.abspath("C:/Users/xoaqu/Documents/SCHOOLSHIT/Master/AAI/Proyecto/waterpotability.csv")

    # Se llama a la función para leer e imputar los datos
    datos, pot = lecturaeimputaciondatos(datapath1)

    # Se realizan los entrenos secuencial y por multihilo y se mide la temporización
    time_sequential = timeit("find_sequential1()", globals=globals(), number=1)
    time_thread = timeit("find_threads1()", globals=globals(), number=1)

    print("La ejecución secuencial ha tardado: ", time_sequential, "segundos.")
    print("La ejecución multihilo ha tardado: ", time_thread, "segundos.")

    find_threads2()
