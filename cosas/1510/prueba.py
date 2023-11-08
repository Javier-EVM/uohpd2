from dataset import loadData
from classifierTree import classifierTree
from classifierNN import classifierNeuralNetwork
from classifierSVM import classifierSVM
from classifierLR import classifierLogisticRegression
from classifierKNN import classifierKNN
from classifier import useClassifier

x,y = loadData("Adult")
x,y = loadData("Banknote")
x,y = loadData("spambase")
#classifierNeuralNetwork(x, y, 0.2,params = "param_defecto",None)
#classifierTree(x, y, 0.2, None)
#classifierLogisticRegression(x, y, 0.2, None)
#classifierSVM(x, y, 0.2,"param_defecto",None)
#classifierKNN(x, y, 0.2, params = "param_defecto")
import time

inicio = time.time()
_,_,_,_,t = useClassifier("Arbol",x,y, params = None)
fin = time.time()
print("hola")
print(fin-inicio)
print(t)



#inicio = time.time()
#useClassifier("NN",x,y, params = None)
#fin = time.time()
#print(fin-inicio)

#inicio = time.time()
#useClassifier("KNN",x,y, params = None)
#fin = time.time()
#print(fin-inicio)

#inicio = time.time()
#useClassifier("LR",x,y, params = None)
#fin = time.time()
#print(fin-inicio)

#inicio = time.time()
#useClassifier("SVM",x,y, params = None)
#fin = time.time()
#print(fin-inicio)