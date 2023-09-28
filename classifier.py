from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from dataset import loadData

def useClassifier(classifier, x ,y ,size = 0.2, params=None, accuracy_solicitado=0.9):
    if classifier == "Arbol":
        a_in,a_out,model,model_params = classifierTree(x,y,size,params)
        return a_in,a_out,model,model_params
    if classifier == "NN":
        a_in,a_out,model,model_params = classifierNeuralNetwork(x, y, size, params)
        return a_in,a_out,model,model_params
    if classifier == "SVM":
        a_in,a_out,model_params,model = classifierSVM(x, y, size, params, accuracy_solicitado)
        return a_in,a_out,model,model_params




def classifierTree(x,y,size,params):
    #Se dividen los datos en Entrenamiento y testeo
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)
    if params == None:
        #Se instancia el clasificador
        clf = DecisionTreeClassifier()

        #Se entrena el clasificador
        clf.fit(X_train, y_train)

        #Se obtienen las predicciones
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        #Se obtiene la accuracy
        accuracy_in = accuracy_score(y_train, y_pred_train)
        accuracy_out = accuracy_score(y_test, y_pred_test)

        #print(f"La accuracy es: {accuracy:.2f}")
        return accuracy_in,accuracy_out,clf,clf.get_params()
    
    elif params == 'param_defecto':
        # Define el espacio de búsqueda de hiperparámetros
        param_dist = {
            'criterion': ['gini', 'entropy'],  # Tipo de criterio
            'splitter': ['best', 'random'],    # Estrategia de división
            'max_depth': np.arange(1, 11),    # Profundidad máxima del árbol
            'min_samples_split': np.arange(2, 11),  # Mínimo de muestras para dividir un nodo
            'min_samples_leaf': np.arange(1, 11)  # Mínimo de muestras en una hoja
            #'ccp_alpha': list(np.linspace(0.0, 0.2, 100))
        }

        # Crea un clasificador de árbol de decisión
        clf = DecisionTreeClassifier(random_state=42)
        print(clf.get_params())
        # Crea un objeto RandomizedSearchCV
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=100, cv=5, random_state=42, scoring='accuracy')

        # Realiza la búsqueda aleatoria en los datos de entrenamiento
        random_search.fit(X_train, y_train)
        
        #Se obtienen las predicciones
        y_pred_train = random_search.predict(X_train)
        y_pred_test = random_search.predict(X_test)

        #Se obtiene la accuracy
        accuracy_in = accuracy_score(y_train, y_pred_train)
        accuracy_out = accuracy_score(y_test, y_pred_test)

        #print(f"La accuracy es: {accuracy:.2f}")
        return accuracy_in,accuracy_out,random_search.best_estimator_,random_search.best_estimator_.get_params()
    
    else:
        diccionario_params = params
        # Crea un clasificador de árbol de decisión
        clf = DecisionTreeClassifier(random_state=42)
        print(clf.get_params())
        # Crea un objeto RandomizedSearchCV
        random_search = RandomizedSearchCV(clf, param_distributions=diccionario_params, n_iter=100, cv=5, random_state=42, scoring='accuracy')

        # Realiza la búsqueda aleatoria en los datos de entrenamiento
        random_search.fit(X_train, y_train)
        
        #Se obtienen las predicciones
        y_pred_train = random_search.predict(X_train)
        y_pred_test = random_search.predict(X_test)

        #Se obtiene la accuracy
        accuracy_in = accuracy_score(y_train, y_pred_train)
        accuracy_out = accuracy_score(y_test, y_pred_test)

        #print(f"La accuracy es: {accuracy:.2f}")
        return accuracy_in,accuracy_out,random_search.best_estimator_,random_search.best_estimator_.get_params()



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

        return accuracy_in, accuracy_out, clf, clf.get_params()

    elif params == 'param_defecto':
        #Define el espacio de búsqueda de hiperparámetros
        param_dist = {
            'hidden_layer_sizes': [(10, 10), (20, 20), (30, 30)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
        }

        #Crea un clasificador de red neuronal
        clf = MLPClassifier(max_iter=1000, random_state=42)

        #Crea un objeto RandomizedSearchCV
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=100, cv=5, random_state=42, scoring='accuracy')

        #Realiza la búsqueda aleatoria en los datos de entrenamiento
        random_search.fit(X_train, y_train)

        #Obtiene las predicciones
        y_pred_train = random_search.predict(X_train)
        y_pred_test = random_search.predict(X_test)

        #Obtiene la accuracy
        accuracy_in = accuracy_score(y_train, y_pred_train)
        accuracy_out = accuracy_score(y_test, y_pred_test)

        return accuracy_in, accuracy_out, random_search.best_estimator_, random_search.best_estimator_.get_params()

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

        return accuracy_in, accuracy_out, random_search.best_estimator_, random_search.best_estimator_.get_params()


def classifierSVM(x, y, samp_size, params, accuracy_solicitado):
    '''
    classifierSVM(x, y, samp_size=0.2, params=None, accuracy_solicitado=0.9):

    x: matriz de atributos
    y: matriz de datos "target"
    sam_size: valor de tamaño de muestra para prueba del clasificador, por defecto tiene valor 0.2
    params: recibe de entrada un diccionario con los parámetros en interés del clasificador
            y una lista de los valores a probar respectivamente.
            Hay una posibilidad de un diccionario por defecto por nombre 'param_defecto', que su valor está dado por

            -kernel: ['linear', 'rbf', 'poly']
            -C: [0.1, 1, 10, 100]
            -gamma: [0.001, 0.01, 0.1, 1]
            -degree: [2, 3, 4, 5]
    ----------------------------------------------------------------------------------------------------------------------
    accuracy_solicitado: valor de accuracy deseado por el usuario, recibe valores tipo "float" de entre 0 y 1,
                         por defecto tiene por valor 0.9
    ----------------------------------------------------------------------------------------------------------------------
    la salida tiene la siguiente estructura: 

    salida (return):  accuracy_in, acc_out, parametros_final, clasificador

            -accuracy_in: Precisión del modelo en el conjunto de entrenamiento.

            -accuracy_out: Precisión del modelo en el conjunto de prueba.

            -parametros_final: Parámetros finales del clasificador.
            
            -clasificador: Este valor es el propio clasificador. Se devuelve para que, principalmente el usuario pueda 
            utilizarlo de manera directa para futuras predicciones adicionales o cualquier otra operación con él.
    '''
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=samp_size, random_state=42)

    if params == None:
        #Entrenar inicialmente con kernel lineal
        svm = SVC(kernel='linear')
        svm.fit(X_train, y_train)
        y_svm = svm.predict(X_test)        
        accuracy = accuracy_score(y_true=y_test, y_pred=y_svm)
        
        #Condicional: Si el accuracy es menor al solicitado, intentar con kernel RBF
        if accuracy < accuracy_solicitado:
            svm = SVC(kernel='rbf')
            
    elif params == 'param_defecto':
        # Se realiza Randomized Search para buscar hiperparámetros
        diccionario_params = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4, 5]}
        random_search = RandomizedSearchCV(estimator=SVC(), param_distributions=diccionario_params, n_iter=10, scoring='accuracy', cv=5, random_state=42)
        random_search.fit(X_train, y_train)
        svm = random_search.best_estimator_
        y_svm = svm.predict(X_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_svm)

    else:
        diccionario_params = params
        random_search = RandomizedSearchCV(estimator=SVC(), param_distributions=diccionario_params, n_iter=10, scoring='accuracy', cv=5, random_state=42)
        random_search.fit(X_train, y_train)
        svm = random_search.best_estimator_

    #Desde svm.fit
    parametros_final = svm.get_params()
    svm.fit(X_train, y_train)

    #Predicciones
    y_svm_train = svm.predict(X_train)
    y_svm_test = svm.predict(X_test)

    #Métricas
    accuracy_in = accuracy_score(y_train, y_svm_train)
    accuracy_out = accuracy_score(y_test, y_svm_test)


    #Return
    if (accuracy_out >= accuracy_solicitado):
        return accuracy_in, accuracy_out, parametros_final, svm
    else:
        print(f'No se ha llegado al accuracy {accuracy_solicitado}, el mejor valor es encontrado por medio de Random Search es:')
        return round(accuracy_in, 3), round(accuracy_out, 3), parametros_final, svm

x,y = loadData("Banknote")

print(useClassifier("Arbol",x,y, params=None))
print(classifierNeuralNetwork(x, y, 0.2, None))
print(classifierSVM(x, y, 0.2, "param_defecto", 0.9))
#print(useClassifier("SVM",x,y, params="param_defecto"))


#print(useClassifier("NN",x,y, params="param_defecto"))
#print(useClassifier("NN",x,y, params={'hidden_layer_sizes': [(10, 10), (20, 20), (30, 30)],'activation': ['relu', 'tanh', 'logistic'],'alpha': [0.0001, 0.001, 0.01, 0.1],}))
