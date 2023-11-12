from dataset import loadData
from classifierTree import classifierTree
from classifierNN import classifierNeuralNetwork
from classifierSVM import classifierSVM
from classifierLR import classifierLogisticRegression
from classifierKNN import classifierKNN
from classifier import useClassifier
from tabulate import tabulate


datasets_names = ['house-votes-84', 'spambase', "Monks1", "Monks3", "Monks3", "Credit-approval"
                  ,"Ionosphere", " ", "Musk-2", "Connectionist-bench", "Breast-cancer", "Pima-diabetes",
                  "Statlog", "Banknote", "Bank-marketing"]


def prueba(clf, repeticiones, names):
    title = [["Dataset",f"{clf}"," "," "," ", " ", " "],[" ","Acc. In %", " ", "Acc. Out %", "", "T (s)", ""],[" ","HD", "BA", "HD", "BA", "HD", "BA" ]]
    for i,name in enumerate(names):
        print(name)
        A_in_std = 0
        A_in_ba = 0
        A_out_std = 0
        A_out_ba = 0
        T_std = 0
        T_ba = 0
        for p in range(repeticiones):
            x,y = loadData(name) 
            a_in,a_out,_,_,t = useClassifier(clf,x,y, params = None)
            A_in_std += a_in
            A_out_std += a_out
            T_std += t

            #Adult no termina en SVM BA
            if (name == "Adult" and clf == "SVM"):
                a_in,a_out,_,_,t = 0,0
                A_in_ba += 0
                A_out_ba += 0
                T_ba += 0
                 

            #else:
            a_in,a_out,_,_,t = useClassifier(clf,x,y, params = "param_defecto")
            A_in_ba += a_in
            A_out_ba += a_out
            T_ba += t
        
        A_in_std = 100*A_in_std/repeticiones
        A_in_ba = 100*A_in_ba/repeticiones
        A_out_std = 100*A_out_std/repeticiones
        A_out_ba = 100*A_out_ba/repeticiones
        T_std = T_std/repeticiones
        T_ba = T_ba/repeticiones
        
        title.append([f"{name}",f'{A_in_std:.2f}',f'{A_in_ba:.2f}',f'{A_out_std:.2f}',f'{A_out_ba:.2f}',f'{T_std:.4f}',f'{T_ba:.4f}'])

    with open(f"Prueba_{clf}.txt", 'w') as f:
            f.write(tabulate(title, headers = "firstrow", tablefmt = "grid"))
            f.write('\n')   
        


prueba("Arbol", 3, ['house-votes-84', 'spambase', "Monks1", "Monks2", "Monks3", "Credit-approval"
                  ,"Ionosphere", "Adult", "Musk-2", "Connectionist-bench", "Breast-cancer", "Pima-diabetes",
                  "Statlog", "Banknote", "Bank-marketing"])

prueba("LR", 3, ['house-votes-84', 'spambase', "Monks1", "Monks2", "Monks3", "Credit-approval"
                  ,"Ionosphere", "Adult", "Musk-2", "Connectionist-bench", "Breast-cancer", "Pima-diabetes",
                  "Statlog", "Banknote", "Bank-marketing"])

prueba("KNN", 3, ['house-votes-84', 'spambase', "Monks1", "Monks2", "Monks3", "Credit-approval"
                  ,"Ionosphere", "Adult", "Musk-2", "Connectionist-bench", "Breast-cancer", "Pima-diabetes",                  "Statlog", "Banknote", "Bank-marketing"])

prueba("SVM", 3, ['house-votes-84', 'spambase', "Monks1", "Monks2", "Monks3", "Credit-approval"
                  ,"Ionosphere", "Adult", "Musk-2", "Connectionist-bench", "Breast-cancer", "Pima-diabetes",
                 "Statlog", "Banknote", "Bank-marketing"])

prueba("NN", 3, ['house-votes-84', 'spambase', "Monks1", "Monks2", "Monks3", "Credit-approval"
                  ,"Ionosphere", "Adult", "Musk-2", "Connectionist-bench", "Breast-cancer", "Pima-diabetes",
                  "Statlog", "Banknote", "Bank-marketing"])