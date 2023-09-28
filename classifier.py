from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from dataset import loadData

def useClassifier(classifier, x ,y ,size = 0.2, params=None, accuracy_solicitado=0.9):
    if classifier == "Arbol":
        a_in,a_out,model,model_params = classifierTree(x,y,size,params)
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
        return accuracy_in,accuracy_out,random_search,random_search.get_params()
    
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
        return accuracy_in,accuracy_out,random_search,random_search.get_params()



x,y = loadData("Banknote")
print(useClassifier("Arbol",x,y, params=None))