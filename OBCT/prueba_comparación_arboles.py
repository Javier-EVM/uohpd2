from heuristica import setMax,setMax3,setMax4
from dataset import loadData
from sklearn.model_selection import train_test_split
from predict import predict
from sklearn.metrics import accuracy_score
from plotOBCT import plotOBCT
from sklearn.metrics import accuracy_score
from classifierTree import classifierTree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from MaxFlowOBCT import MFOBCT

d = 4
x,y = loadData("house-votes-84") 
x,y = loadData("Monks1") 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lams = [0, 0.1, 0.5]
for la in lams:
    #b, w, tiempo, gap = setMax4(x_train, y_train, d , la, 0)
    tiempo, of , b , w , z, gap = MFOBCT(x_train, y_train ,d ,la , False, False, False)
    y_pred_train = predict(b, w, x_train, d)
    y_pred_test = predict(b, w, x_test, d)

    accuracy_in = accuracy_score(y_train, y_pred_train)
    accuracy_out = accuracy_score(y_test, y_pred_test)

    print(f"----OBCT lambda {la} d {d}---- \n")
    print(f"Acc. In: {accuracy_in}")
    print(f"Acc. Out: {accuracy_out}")
    print(f"Tiempo: {tiempo} \n")
    plotOBCT("",b,w,la,d)

for la in lams:
    b, w, tiempo, gap = setMax4(x_train, y_train, d , la, 0)
    y_pred_train = predict(b, w, x_train, d)
    y_pred_test = predict(b, w, x_test, d)

    accuracy_in = accuracy_score(y_train, y_pred_train)
    accuracy_out = accuracy_score(y_test, y_pred_test)

    print(f"----OBCT Heuristica lambda {la} d {d}---- \n")
    print(f"Acc. In: {accuracy_in}")
    print(f"Acc. Out: {accuracy_out}")
    print(f"Tiempo: {tiempo} \n")
    plotOBCT("Heuristica",b,w,la,d)


print(f"Sklearn d {d} \n")
#SKlearn
clf = DecisionTreeClassifier(max_depth =  3)

#Se entrena el clasificador
clf.fit(x_train, y_train)

#Se obtienen las predicciones
y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)

#Se obtiene la accuracy
accuracy_in = accuracy_score(y_train, y_pred_train)
accuracy_out = accuracy_score(y_test, y_pred_test)

print(f"Acc. In: {accuracy_in}")
print(f"Acc. Out: {accuracy_out}")



feature_names = ['Pos x','Pos y']
labels = ["0","1"]


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
                   class_names = labels,
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
plt.savefig("SKlearn_tree_{d}.png", dpi= 100)

