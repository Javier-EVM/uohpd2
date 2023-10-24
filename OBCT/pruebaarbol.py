from heuristica import setMax,setMax3,setMax4
from dataset import loadData
from sklearn.model_selection import train_test_split
from predict import predict
from sklearn.metrics import accuracy_score
from classifierTree import classifierTree
from sklearn.tree import DecisionTreeClassifier
#b : Nodos brancheo
#w : Nodos clasificación


x,y = loadData("Monks1") 

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x[0:100, :]
x_test = x[101:, :]
y_train = y[0:100]
y_test = y[101:]

clf = DecisionTreeClassifier(max_depth =  3)

#Se entrena el clasificador
clf.fit(x_train, y_train)

#Se obtienen las predicciones
y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)

#Se obtiene la accuracy
accuracy_in = accuracy_score(y_train, y_pred_train)
accuracy_out = accuracy_score(y_test, y_pred_test)

print(accuracy_in)
print(accuracy_out)



import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
feature_names = ['Pos x','Pos y']
labels = ["1","0"]
print(sum(y_train))
print(len(y_train))
print(x)

#X = [1. 0. 0. 1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 1.]
#0 a 14
#print(clf.predict(X))
#0 a 14
#plt the figure, setting a black background
plt.figure(figsize=(10,5), facecolor ='w')
#create the tree plot

a = tree.plot_tree(clf,
                   #use the feature names stored
                   #feature_names = feature_names,
                   #use the class names stored
                   class_names = True,
                   rounded = True,
                   filled = True,
                   fontsize=10)
#show the plot

#Value [0,1]
#Hay 51 unos en total
#Representa cantidad de etiqueta con 0 y cantidad de etiqueta con 1
#SI ir izquierda
#No ir derecha
plt.show()
plt.savefig("Salida-arbol.png", dpi= 100)