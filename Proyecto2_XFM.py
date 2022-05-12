# Código por Xoaquin Fabeiro Monteagudo

# Importación de librerías
from statistics import mode
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold, chi2
from sklearn.feature_selection import SelectKBest
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from sympy import im
from tensorflow import keras
from keras import layers, models
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA, FastICA
import warnings

# Función de creación de modelo con separación de datos de entrenamiento y test
def ModeloSeparacionTestTrain(
    modelo,
    claveModelo,
    data,
    out,
    porcentajetrain,
    CV=10,
    scoring=["accuracy", "recall", "precision", "f1", "roc_auc", "neg_mean_squared_error"],
):
    # Se realiza una separación de los datos en dos subconjuntos, uno de entrenamiento y otro de test.
    Train, Test, t_train, t_test = train_test_split(data, out, train_size=porcentajetrain)

    print("El conjunto de datos de entrenamiento", claveModelo, "contiene", len(Train), "datos.")
    print("El conjunto de datos de test", claveModelo, "contiene", len(Test), "datos.")

    modelo.fit(Train, t_train)

    # Se obtiene las salida del modelo entrenado
    y_test = modelo.predict(Test)

    # Se calcula el número de aciertos totales del modelo.
    TruesTest = np.sum((y_test == t_test), axis=0)
    print("El conjunto de test", claveModelo, "ha tenido ", TruesTest, " aciertos de", len(Test), ".")

    # Se realiza una validación cruzada
    scores = cross_validate(modelo, Test, t_test, cv=CV, scoring=scoring)
    scoresm = {}
    for clave in scores:
        scores[clave]
        scoresm[clave] = np.mean(scores[clave])
    return scores, scoresm


# Función para la lectura e imputación de los datos.
def lecturaeimputaciondatos(path):
    # Lectura y desordenación de los datos.
    datos = pd.read_csv(path)
    datos = datos.sample(frac=1).reset_index(drop=True)

    # Se hace la imputación de datos para eliminar datos tipo NaN utilizando una imputacion de tipo KNN.
    impK = KNNImputer(missing_values=np.nan, n_neighbors=15)
    datos = impK.fit_transform(datos)

    # Se separan los datos en las características y la salida, que es la potabilidad.
    out = datos[:, -1]
    datos = datos[:, :-1]

    return datos, out


# Función para crear un modelo utilizando redes neuronales basadas en keras
def modeloRN(neurxcapa, lr, n_inputs, metrics=["accuracy", "Recall", "Precision", "AUC"]):
    modelo = models.Sequential()
    # En función del tamaño del vector de neurx capa, que almacena las neuronas por capa, se crea la red.
    for capa in range(1, len(neurxcapa) + 1):
        if capa == 1:
            modelo.add(layers.Dense(neurxcapa[0], input_dim=n_inputs, activation="relu"))
        elif capa == len(neurxcapa):
            modelo.add(layers.Dense(2, activation="softmax"))
        else:
            modelo.add(layers.Dense(neurxcapa[capa], activation="relu"))

    # Se apliica el optimizador Adam y se compila el modelo
    opt = keras.optimizers.Adam(learning_rate=lr)
    modelo.compile(loss="categorical_crossentropy", optimizer=opt, metrics=metrics)

    return modelo


# Función para entrenar el modelo de red de neurons sin validación cruzada
def entrenoRN(model, data, out, porcentajetrain):
    # Se categoriza la salida en los dos tipos de clases posibles (potable y no potable)
    out_cat = to_categorical(out, num_classes=2)

    # Se particiona el conjunto de datos en entrenamiento y test
    Train, Test, t_train, t_test = train_test_split(data, out_cat, train_size=porcentajetrain)
    # Se realiza el entrenamiento del modelo
    history = model.fit(Train, t_train, validation_data=(Test, t_test), epochs=300, batch_size=40, verbose=0)

    # Se dibuja la gráfica con la evolución de la exactitud en función de las iteraciones
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Exactitud del modelo")
    plt.ylabel("Exactitud (Accuracy)")
    plt.xlabel("Iteración (epoch)")
    plt.legend(["Entrenamiento", "Test"], loc="lower right")
    plt.show()

    # Se calculan las métricas de evaluación
    loss_and_metrics = model.evaluate(Test, t_test, batch_size=None)

    # Se obtienen las salidas del modelo entrenado
    y = model.predict(Test)
    for salida in y:
        if salida[0] >= 0.5:
            salida[0] = 1
            salida[1] = 0
        else:
            salida[0] = 0
            salida[1] = 1
    TruesTest = np.sum((y == t_test), axis=0)

    print("El conjunto de test de la red neuronal ha tenido ", TruesTest[0], " aciertos de", len(Test), ".")

    return loss_and_metrics


# Se define la función de entrenamiento de red de neuronas con validación cruzada
def ValidacionCruzadaRN(data, out, metrics, k_folds=5, k_fold_reps=2, epochs=300, batch_size=40):
    n, m = data.shape
    out_cat = to_categorical(out, num_classes=2)

    # Se crea un data dataframe de pandas para guardar los resultados de las métricas de evaluación
    results = pd.DataFrame(columns=metrics)

    # Se genera una K-fold estratificada
    rkf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=k_fold_reps, random_state=42)

    # Se realizan tantos entrenamientos como valor de se indica en la validación cruzada
    for i, (train_index, test_index) in enumerate(rkf.split(data, out)):
        # Muestra el paso de la k-fold en la que nos encontramos
        print("k_fold", i + 1, "de", k_folds * k_fold_reps)

        # Se obtienen los paquetes de datos de entrenamiento y test en base a los índices aleatorios generados en la k-fold
        X_train, t_train = data[train_index], out_cat[train_index]
        X_test, t_test = data[test_index], out_cat[test_index]

        # Se carga el modelo en cada paso de la kfold para resetear el entrenamiento (pesos)
        model = modeloRN([5, 14, 2], 0.002, m, metrics)

        # Se realiza el entrenamiento de la red de neuronas
        history = model.fit(
            X_train, t_train, validation_data=(X_test, t_test), epochs=epochs, batch_size=batch_size, verbose=0
        )

        # Se añade una línea en la tabla de resultados (dataframe de pandas) con los resultados de las métricas seleccionadas
        results.loc[i] = model.evaluate(X_test, t_test, batch_size=None)[
            1:
        ]  # Se descarta la métrica 0 porque es el valor de la función de error

    for metric in metrics:
        print(metric.upper(), "(media):", np.mean(results[metric].values))


# Función para dibujar el diagrama de cajas
def dibujacajas(modelos, Labs, scores):
    # Se definen las etiquetas, los objetos y se dibuja la gráfica
    for score in scores.columns:
        fig, ax = plt.subplots()
        ax.set_title(score)
        ax.boxplot(scores[score], labels=Labs)
        plt.show()


# Función para realizar el contraste de hipótesis
def contrastehip(modelos, test, CV=10):
    # Se realiza el contraste de hiótesis
    lab = [lab[0] for lab in modelos]
    alpha = 0.05
    F_statistic, pVal = stats.kruskal(*test)
    print("p-valor KrusW:", pVal)
    if pVal <= alpha:
        print("Rechazamos la hipótesis: los modelos son diferentes\n")
        stacked_data = np.vstack(test).ravel()
        stacked_model = np.reshape([np.repeat(lab[0], CV) for lab in modelos], (1, CV * len(lab))).ravel()
        MultiComp = MultiComparison(stacked_data, stacked_model)
        print(MultiComp.tukeyhsd(alpha=0.05))
    else:
        print("Aceptamos la hipótesis: los modelos son iguales")


def main():
    warnings.filterwarnings("ignore")
    # Se define el path del conjunto de datos
    datapath1 = os.path.abspath("D:/Universitas/Master_IIR/Master/AA2/Proyecto/waterpotability.csv")
    datapath2 = os.path.abspath("C:/Users/xoaqu/Documents/SCHOOLSHIT/Master/AAI/Proyecto/waterpotability.csv")

    # Se llama a la función para leer e imputar los datos
    datos, pot = lecturaeimputaciondatos(datapath1)
    # Se definen los modelos en base a los mejores parámetros encontrados
    modeloLR = make_pipeline(
        StandardScaler(),
        PCA(n_components=6),
        LogisticRegression(penalty="none", solver="lbfgs", max_iter=7000, multi_class="multinomial"),
    )
    modeloKNN = make_pipeline(StandardScaler(), FastICA(n_components=9), KNeighborsClassifier(n_neighbors=27))
    modeloDT = make_pipeline(
        StandardScaler(), FastICA(n_components=9), DecisionTreeClassifier(criterion="entropy", max_depth=7)
    )

    modelos = [modeloLR, modeloKNN, modeloDT]
    # Entrenamientos de los modelos
    print("Entrenamiento de los modelos:")
    scoresLR, scoresMLR = ModeloSeparacionTestTrain(modelos[0], "LR", datos, pot, 0.8)
    scoresKNN, scoresMKNN = ModeloSeparacionTestTrain(modelos[1], "KNN", datos, pot, 0.8)
    scoresDT, scoresMDT = ModeloSeparacionTestTrain(modelos[2], "DT", datos, pot, 0.8)

    print("Resultados del entrenamiento de los modelos:")
    print(pd.DataFrame([scoresMLR, scoresMKNN, scoresMDT], index=["LR", "KNN", "DT"]))

    scores = [scoresLR, scoresKNN, scoresDT]
    scoresdf = pd.DataFrame(scores)

    dibujacajas(modelos, ["LR", "KNN", "DT"], scoresdf)

    # contrastehip(modelos, [scoresLR["test_accuracy"],scoresKNN['test_accuracy'], scoresDT['test_accuracy']])

    # Uso de redes de neuronas
    print("Entrenamiento con Redes Neuronales:")

    new_datos = make_pipeline(StandardScaler(), PCA(n_components=5)).fit_transform(datos)
    n, m = new_datos.shape
    metrics = ["accuracy", "AUC", "Recall", "Precision", "mean_squared_error"]
    modeloRed = modeloRN([5, 14, 2], 0.002, m, metrics)
    modeloRed.summary()

    print("Entrenamiento del modelo sin validación cruzada:")
    entrenoRN(modeloRed, new_datos, pot, 0.8)

    print("Entrenamiento de modelo con validación cruzada:")
    ValidacionCruzadaRN(new_datos, pot, metrics)


if __name__ == "__main__":
    main()
