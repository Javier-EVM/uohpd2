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
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None] + list(np.arange(1, 21)),
    'min_samples_split': list(range(2, 21)),
    'min_samples_leaf': list(range(1, 21)),
    'ccp_alpha': list(np.linspace(0.0, 0.2, 100)),
    'class_weight': [None, 'balanced'],
    'max_features': [None, 'sqrt', 'log2'],
    'max_leaf_nodes': [None] + list(range(2, 101)),
    'min_impurity_decrease': list(np.linspace(0.0, 0.2, 100)),
    'min_weight_fraction_leaf': list(np.linspace(0.0, 0.5, 100)),
    'random_state': [42],
}

# Crea un clasificador de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# Crea un objeto RandomizedSearchCV
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, cv=5, random_state=42, scoring='accuracy')

# Supongamos que tienes datos de entrenamiento (X_train, y_train)
# Realiza la búsqueda aleatoria en los datos de entrenamiento
random_search.fit(X_train, y_train)

# Obtiene los 10 mejores resultados
results = random_search.cv_results_
indices = np.argsort(results['mean_test_score'])[::-1][:10]

print("Top 10 Mejores Hiperparámetros:")
for index in indices:
    print(f"Accuracy: {results['mean_test_score'][index]:.4f}")
    print("Hiperparámetros:")
    for param, value in results['params'][index].items():
        print(f"{param}: {value}")
    print("\n")