from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def useClassifier(classifier,x,y,size = 0.2):
    if classifier == "Arbol":
        a_in,a_out,model = classifierTree(x,y,size)
        return a_in,a_out,model




def classifierTree(x,y,size):
    #Se dividen los datos en Entrenamiento y testeo
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)
    print(X_train)
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
    return accuracy_in,accuracy_out,clf