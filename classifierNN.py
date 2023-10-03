
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def classifierNeuralNetwork(x, y, size, params):
    #Se dividen los datos en entrenamiento y testeo
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)

    if params is None:
        #Se instancia el clasificador de red neuronal
        clf = MLPClassifier()

        #Se entrena el clasificador
        clf.fit(X_train, y_train)

        #Se obtienen las predicciones
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        #Se obtiene la accuracy
        accuracy_in = accuracy_score(y_train, y_pred_train)
        accuracy_out = accuracy_score(y_test, y_pred_test)

        return accuracy_in, accuracy_out, clf.get_params(), clf

    elif params == 'param_defecto':
        #Define el espacio de búsqueda de hiperparámetros
        param_dist = {
            'hidden_layer_sizes': [(10, 10), (20, 20), (30, 30)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.001, 0.01, 0.1],
        }

        #Crea un clasificador de red neuronal
        clf = MLPClassifier(max_iter=500, random_state=42)

        #Crea un objeto RandomizedSearchCV
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, scoring='accuracy')

        #Realiza la búsqueda aleatoria en los datos de entrenamiento
        random_search.fit(X_train, y_train)

        #Obtiene las predicciones
        y_pred_train = random_search.best_estimator_.predict(X_train)
        y_pred_test = random_search.best_estimator_.predict(X_test)

        #Obtiene la accuracy
        accuracy_in = accuracy_score(y_train, y_pred_train)
        accuracy_out = accuracy_score(y_test, y_pred_test)

        return accuracy_in, accuracy_out, random_search.best_estimator_.get_params(), random_search.best_estimator_

    else:
        diccionario_params = params

        #Se crea un clasificador de red neuronal
        clf = MLPClassifier()

        #Se crea un objeto RandomizedSearchCV con parámetros personalizados
        random_search = RandomizedSearchCV(clf, param_distributions=diccionario_params, n_iter=100, cv=5, random_state=42, scoring='accuracy')

        #Realiza la búsqueda aleatoria en los datos de entrenamiento
        random_search.fit(X_train, y_train)

        #Obtiene las predicciones
        y_pred_train = random_search.predict(X_train)
        y_pred_test = random_search.predict(X_test)

        #Obtiene la accuracy
        accuracy_in = accuracy_score(y_train, y_pred_train)
        accuracy_out = accuracy_score(y_test, y_pred_test)

        return accuracy_in, accuracy_out, random_search.best_estimator_.get_params(), random_search.best_estimator_