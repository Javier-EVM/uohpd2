from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from dataset import loadData
import numpy as np

#x,y = loadData('house-votes-84')
#x,y = loadData('spambase')
#x,y = loadData("Credit-approval")
x,y = loadData("Adult")
x,y = loadData("Connectionist-bench")
x,y = loadData("Breast-cancer")
x,y = loadData("Pima-diabetes")
x,y = loadData("Banknote")
#Se dividen los datos en Entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)




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

# Supongamos que tienes datos de entrenamiento (X_train, y_train)
# Realiza la búsqueda aleatoria en los datos de entrenamiento
random_search.fit(X_train, y_train)

# Obtén los resultados de la búsqueda y ordénalos por precisión en orden descendente
results = random_search.cv_results_
indices = np.argsort(results['mean_test_score'])[::-1]

# Muestra el "top 10" de los mejores hiperparámetros y sus puntuaciones de precisión
top_10_params = [(results['params'][i], results['mean_test_score'][i]) for i in indices[:10]]
for i, (params, score) in enumerate(top_10_params, 1):
    print(f"Top {i}: Precisión = {score:.4f}, Hiperparámetros = {params}")