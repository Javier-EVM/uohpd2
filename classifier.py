from dataset import loadData
from classifierTree import classifierTree
from classifierNN import classifierNeuralNetwork
from classifierSVM import classifierSVM
from classifierLR import classifierLogisticRegression
from classifierKNN import classifierKNN



def useClassifier(classifier, x ,y ,size = 0.2, params=None, accuracy_solicitado=0.9):
    if classifier == "Arbol":
        a_in,a_out,model,model_params = classifierTree(x,y,size,params)
        print(f'\n\nAccuracy in: {round(a_in,3)}\n\nAccuracy out: {round(a_out,3)}\n\nParámetros finales: {model_params}\n\n\n')
        return a_in,a_out,model_params,model
    if classifier == "NN":
        a_in,a_out,model,model_params = classifierNeuralNetwork(x, y, size, params)
        print(f'\n\nAccuracy in: {round(a_in,3)}\n\nAccuracy out: {round(a_out,3)}\n\nParámetros finales: {model_params}\n\n\n')
        return a_in,a_out,model_params,model
    if classifier == "SVM":
        a_in,a_out,model_params,model = classifierSVM(x, y, size, params, accuracy_solicitado)
        print(f'\n\nAccuracy in: {round(a_in,3)}\n\nAccuracy out: {round(a_out,3)}\n\nParámetros finales: {model_params}\n\n\n')
        return a_in,a_out,model,model_params
    if classifier == "LR":
        a_in,a_out,model_params,model = classifierLogisticRegression(x, y, size, params, accuracy_solicitado)
        print(f'\n\nAccuracy in: {round(a_in,3)}\n\nAccuracy out: {round(a_out,3)}\n\nParámetros finales: {model_params}\n\n\n')
        return a_in,a_out,model,model_params
    if classifier == "KNN":
        a_in,a_out,model_params,model = classifierKNN(x, y, size, params, accuracy_solicitado)
        print(f'\n\nAccuracy in: {round(a_in,3)}\n\nAccuracy out: {round(a_out,3)}\n\nParámetros finales: {model_params}\n\n\n')
        return a_in,a_out,model,model_params


x,y = loadData("Banknote")

#print(useClassifier("Arbol",x,y, params = 'param_defecto'))
#print(classifierNeuralNetwork(x, y, 0.2, None))
#print(classifierSVM(x, y, 0.2,"param_defecto", 0.9))
#print(useClassifier("KNN",x,y, params = None))
useClassifier("LR",x,y, params = 'param_defecto1')

#print(useClassifier("SVM",x,y, params="param_defecto"))


#print(useClassifier("NN",x,y, params="param_defecto"))
#print(useClassifier("NN",x,y, params={'hidden_layer_sizes': [(10, 10), (20, 20), (30, 30)],'activation': ['relu', 'tanh', 'logistic'],'alpha': [0.0001, 0.001, 0.01, 0.1],}))
