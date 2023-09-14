from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from dataset import loadData

x,y = loadData('house-votes-84')
#x,y = loadData('spambase')

#Se dividen los datos en Entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(X_train)
#Se instancia el clasificador
clf = DecisionTreeClassifier()

#Se entrena el clasificador
clf.fit(X_train, y_train)

#Se obtienen las predicciones
y_pred = clf.predict(X_test)

#Se obtiene la accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"La accuracy es: {accuracy:.2f}")

