from dataset import loadData
from classifierTree import classifierTree
from classifierNN import classifierNeuralNetwork
from classifierSVM import classifierSVM
from classifierLR import classifierLogisticRegression
from classifierKNN import classifierKNN

x,y = loadData("Adult")

print(classifierNeuralNetwork(x, y, 0.2, None))
print(classifierTree(x, y, 0.2, None))
print(classifierLogisticRegression(x, y, 0.2, None))
print(classifierSVM(x, y, 0.2,"param_defecto",None))


print(classifierKNN(x, y, 0.2, params = None))

