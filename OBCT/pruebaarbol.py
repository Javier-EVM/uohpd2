from heuristica import setMax,setMax3,setMax4
from dataset import loadData
from sklearn.model_selection import train_test_split
from predict import predict
from sklearn.metrics import accuracy_score
from classifierTree import classifierTree
#b : Nodos brancheo
#w : Nodos clasificación


x,y = loadData("Monks1") 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#b, w, tiempo = setMax(x_train, y_train, 3 , 0)
accuracy_in, accuracy_out, params, clf = classifierTree(x,y,0.2,None)


print(accuracy_in)
print(accuracy_out)



import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
feature_names = ['Pos x','Pos y']
labels = ['1','0']

#plt the figure, setting a black background
plt.figure(figsize=(30,10), facecolor ='w')
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
plt.show()