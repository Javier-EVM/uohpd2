from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dataset import loadData

x,y = loadData('house-votes-84')
#x,y = loadData('spambase')
x,y = loadData("Adult")
#Se dividen los datos en Entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Crea un modelo de red neuronal
model = MLPClassifier(hidden_layer_sizes=(50), max_iter=1000, random_state=42)

# Entrena el modelo
model.fit(X_train, y_train)

#Se obtienen las predicciones
y_pred = model.predict(X_test)

#Se obtiene la accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"La accuracy es: {accuracy:.2f}")

