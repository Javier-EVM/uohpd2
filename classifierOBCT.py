from dataset2 import loadData
from tabulate import tabulate
from heuristica import setMax,setMax3,setMax4
from predict import predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from MaxFlowOBCT import MFOBCT

def classifierOBCT(x,y,size,params):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)

    if params == None:
        d = 4
        la = 0
        #Entrenar inicialmente con kernel lineal
        #d, lambda, heu
        #b, w, tiempo, gap = setMax4(X_train, y_train, 4 , 0, 0)
        tiempo, of , b , w , z, gap = MFOBCT(X_train, y_train ,d ,la , False, False, False)
        y_pred_train = predict(b, w, X_train, d)
        y_pred_test = predict(b, w, X_test, d)

    else:
        return 0, 0, [], []

    """
    elif params == 'param_defecto':
        # Se realiza Randomized Search para buscar hiperparámetros
        diccionario_params = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4, 5]}
        random_search = RandomizedSearchCV(estimator=SVC(), param_distributions=diccionario_params, n_iter=5, scoring='accuracy', cv=2, random_state=42)
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

    

    """
    #Métricas
    accuracy_in = accuracy_score(y_train, y_pred_train)
    accuracy_out = accuracy_score(y_test, y_pred_test)

    return accuracy_in, accuracy_out, [], []
