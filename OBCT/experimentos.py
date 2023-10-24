from dataset import loadData
from classifierTree import classifierTree
from classifier import useClassifier
from tabulate import tabulate
from heuristica import setMax,setMax3,setMax4
from predict import predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prueba(clf, repeticiones, names):
    title = [["Dataset",f"{clf}"," "," "," ", " ", " "," ", " ", " "],[" ","Acc. In %", " ", " ", "Acc. Out %", "", " ", "T (s)", "", " "],[" ","HD", "BA","OCBT", "HD", "BA","OCBT", "HD", "BA","OCBT" ]]
    #title = [["Dataset",f"{clf}"," "," "," ", " ", " "],[" ","Acc. In %", " ", "Acc. Out %", "", "T (s)", ""],[" ","HD", "BA", "HD", "BA", "HD", "BA" ]]
    with open("OBCT.txt","w") as r:
        for i,name in enumerate(names):
            print(name)
            A_in_std = 0
            A_in_ba = 0
            A_out_std = 0
            A_out_ba = 0
            T_std = 0
            T_ba = 0

            A_in_OBCT = 0
            A_out_OBCT = 0
            T_OBCT = 0

            for p in range(repeticiones):
                x,y = loadData(name) 
                a_in,a_out,_,_,t = useClassifier(clf,x,y, params = None)
                A_in_std += a_in
                A_out_std += a_out
                T_std += t
                    
                a_in,a_out,_,_,t = useClassifier(clf,x,y, params = "param_defecto")
                A_in_ba += a_in
                A_out_ba += a_out
                T_ba += t

                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                #b, w, tiempo = setMax(x_train, y_train, 3 , 0)
                b, w, tiempo, gap = setMax4(x_train, y_train, 5 , 0, 2)
                r.write(f"Gap {name} {p}: {gap}  \n")
                y_pred_train = predict(b, w, x_train, 5)
                y_pred_test = predict(b, w, x_test, 5)

                A_in_OBCT += accuracy_score(y_train, y_pred_train)
                A_out_OBCT += accuracy_score(y_test, y_pred_test)
                T_OBCT += tiempo


            
            A_in_std = 100*A_in_std/repeticiones

            A_in_ba = 100*A_in_ba/repeticiones
            A_out_std = 100*A_out_std/repeticiones
            A_out_ba = 100*A_out_ba/repeticiones
            T_std = T_std/repeticiones
            T_ba = T_ba/repeticiones

            A_in_OBCT = 100*A_in_OBCT/repeticiones
            A_out_OBCT = 100*A_out_OBCT/repeticiones
            T_OBCT = T_OBCT/repeticiones
            
            title.append([f"{name}",f'{A_in_std:.2f}',f'{A_in_ba:.2f}',f'{A_in_OBCT:.2f}',f'{A_out_std:.2f}',f'{A_out_ba:.2f}',f'{A_out_OBCT:.2f}',f'{T_std:.4f}',f'{T_ba:.4f}',f'{T_OBCT:.4f}'])

        with open(f"Prueba_{clf}-OBCT.txt", 'w') as f:
                f.write(tabulate(title, headers = "firstrow", tablefmt = "grid"))
                f.write('\n')  

datasets_names = ['house-votes-84', "Monks1", "Monks2", "Monks3"]

prueba("Arbol", 3, datasets_names)


"""
def pruebaOBCT(repeticiones, names, clf = "OBCT"):
    title = [["Dataset",f"{clf}"," "," "," ", " ", " "," ", " ", " "],[" ","Acc. In %", " ", " ", "Acc. Out %", "", " ", "T (s)", "", " "],[" ","HD", "BA","OCBT", "HD", "BA","OCBT", "HD", "BA","OCBT" ]]
    with open("OBCT.txt","w") as r:
              
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
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                #b, w, tiempo = setMax(x_train, y_train, 3 , 0)
                b, w, tiempo, gap = setMax4(x_train, y_train, 5 , 0, 2)

                r.write(f"Gap {name} {p}: {gap} \n")
                y_pred_train = predict(b, w, x_train, 5)
                y_pred_test = predict(b, w, x_test, 5)

                A_in_std += accuracy_score(y_train, y_pred_train)
                A_out_std += accuracy_score(y_test, y_pred_test)
                T_std += tiempo
                    
            
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

""" 

