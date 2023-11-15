from heuristica import setMax, setMax3, setMax4
from dataset2 import loadData
from sklearn.model_selection import train_test_split
from predict import predict
from sklearn.metrics import accuracy_score
from MaxFlowOBCT import MFOBCT
import time
# Hiperparámetros
# d = 4  # Profundidad del árbol
# lambda_ = 0

# Carga de datos
# x, y = loadData("house-votes-84", binary = True)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

n = 1  # Cantidad de nodos a fijar con preguntas que no son "buenas" pero rápidas
# Cuando n = 1, entonces son 2 preguntas
# n = 0 ==> 1 pregunta

# Heurística donde resuelve una relajación y luego el MIP
# b, w, tiempo, gap = setMax4(x_train, y_train, d, lambda_, n)  # OBCT con heurística

# y_pred_train = predict(b, w, x_train, d)
# y_pred_test = predict(b, w, x_test, d)

# accuracy_in = accuracy_score(y_train, y_pred_train)
# accuracy_out = accuracy_score(y_test, y_pred_test)

# print(accuracy_in)
# print(accuracy_out)
# print(tiempo)

# with open(f"Salida-OBCT.txt", 'w') as f:
#     f.write("B")
#     for i in b:
#         f.write(f"b({i}): {b[i]}\n")

#     f.write('\n')
#     f.write("W")
#     for j in w:
#         f.write(f"w({j}): {w[j]}\n")
#     f.write('\n')

# OBCT sin heurística
# tiempo, of, b, w, z, gap = MFOBCT(x_train, y_train, d, lambda_, False, False, False)

# y_pred_train = predict(b, w, x_train, d)
# y_pred_test = predict(b, w, x_test, d)

# accuracy_in = accuracy_score(y_train, y_pred_train)
# accuracy_out = accuracy_score(y_test, y_pred_test)

# print(accuracy_in)
# print(accuracy_out)
# print(tiempo)

# with open(f"Salida-OBCT.txt", 'w') as f:
#     f.write("B")
#     for i in b:
#         f.write(f"b({i}): {b[i]}\n")

#     f.write('\n')
#     f.write("W")
#     for j in w:
#         f.write(f"w({j}): {w[j]}\n")
#     f.write('\n')


    # Qué haría Javier?: 
    # 1) fijar d (profundidad) por ejemplo d = 4
    # 2) para la heuristica, partir con n = 0, si demora mucho, hacer n = 1
    # 3) dejar lambda = 0

def cross_val(clasificador, data,k):
     from sklearn.model_selection import KFold
     data = data.tolist()
     kf = KFold(n_splits=k)
     for train_index, test_index in kf.split(data):
         train_data = [data[i] for i in train_index]
         test_data = [data[i] for i in test_index]


data_names = [
    'house-votes-84', #Sí
    'spambase',#Sí
    'Monks1',#Sí
    'Monks2',#Sí
    'Monks3',#Sí
    'Credit-approval',#Sí
    'Ionosphere',#Sí
    'Adult', #no resuelve para este caso #MODIFICADO 
     'Musk-2', #no resuelve para este caso #MODIFICADO
    'Connectionist-bench', # Sí
    'Breast-cancer', # Sí
     'Pima-diabetes', #no resuelve para este caso #MODIFICADO
     'Statlog', #no resuelve para este caso #MODIFICADO
     'Banknote', #no resuelve para este caso #MODIFICADO
    'Bank-marketing' # Sí
]
data_names1 = [
    # 'house-votes-84', #Sí
    # 'spambase',#Sí
    # 'Monks1',#Sí
    # 'Monks2',#Sí
    # 'Monks3',#Sí
    # 'Credit-approval',#Sí
    # 'Ionosphere',#Sí
    'Adult', #no resuelve para este caso #MODIFICADO 
     'Musk-2', #no resuelve para este caso #MODIFICADO
    # 'Connectionist-bench', # Sí
    # 'Breast-cancer', # Sí
     'Pima-diabetes', #no resuelve para este caso #MODIFICADO #Sí
     'Statlog', #no resuelve para este caso #MODIFICADO #Sí
     'Banknote', #no resuelve para este caso #MODIFICADO #Sí
    # 'Bank-marketing' # Sí
]



# def try_setMax4(x_train, y_train, d, lambda_, n):
#     try:
#         tiempo_inicio = time.time()
#         b, w, tiempo, gap = setMax4(x_train, y_train, d, lambda_, n)
#         print('b: ',b)
#         tiempo_fin = time.time()
#         y_pred_train = predict(b, w, x_train, d)
#         y_pred_test = predict(b, w, x_test, d)
#         accuracy_in = accuracy_score(y_train, y_pred_train)
#         accuracy_out = accuracy_score(y_test, y_pred_test)
#         return accuracy_out, tiempo_fin - tiempo_inicio
#     except Exception as e:
#         print(f"Error al ejecutar con n = {n} y lambda = {lambda_}: {e}")
#         return None

# for name in data_names1:
#     x, y = loadData(name, binary=True)
#     print('\n\n', name, '\n')
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#     d = 4
#     n = 0  # Cantidad de nodos a fijar con preguntas que no son "buenas" pero rápidas
#     # Cuando n = 1, entonces son 2 preguntas
#     # n = 0 ==> 1 pregunta
#     max_n = 1  # Valor máximo de n

#     for n in range(n, max_n + 1):
#         for lambda_ in [0 , 0.1, 0.5]:
#             lambda_ /= 10.0  # Convertir a valores decimales de 0.1 a 0.9
#             result = try_setMax4(x_train, y_train, d, lambda_, n)
#             if result is not None:
#                 accuracy_out, tiempo = result
#                 print(f'n = {n}, lambda = {lambda_}, accuracy: {accuracy_out}, tiempo: {tiempo}')
#                 break  # Salir del bucle si la ejecución fue exitosa

#         break


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)




def cross_val_obct(data, k):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k)
    data_x,data_y =  loadData(data, binary = True)
    d = 4
    n = 0
    lambda_ = 0
    accuracies = []  # Lista para almacenar las precisiones de cada división
    for train_index, test_index in kf.split(data_x):
        if data[0:5] == 'Monks':
            x_train, y_train = data_x[train_index, :], data_y[train_index, :]
            x_test, y_test = data_x[test_index, :], data_y[test_index, :]
        else:
            x_train, y_train = data_x[train_index, :],data_y[train_index]
            x_test, y_test = data_x[test_index, :],data_y[test_index]
        b, w, tiempo, gap = setMax4(x_train, y_train, d, lambda_, n) 
        y_pred_test = predict(b, w, x_test, d)
        accuracy = accuracy_score(y_test, y_pred_test)
        accuracies.append(accuracy)  # Almacena la precisión de esta división

    mean_accuracy = sum(accuracies) / len(accuracies)  # Precisión promedio

    return accuracies, mean_accuracy
# x,y = loadData('house-votes-84', binary = True)
# print(cross_val_obct(x,y,5))
lista_datasets = [
    #'house-votes-84', #Sí
    #'spambase',#Sí
    #'Monks1',#Sí
    #'Monks2',#Sí
    #'Monks3',#Sí
    #'Credit-approval',#Sí
    #'Ionosphere',#Sí
    # 'Adult', #no resuelve para este caso #MODIFICADO 
    #  'Musk-2', #no resuelve para este caso #MODIFICADO
    #'Connectionist-bench', # Sí
    #'Breast-cancer', # Sí
     'Pima-diabetes', #no resuelve para este caso #MODIFICADO
     'Statlog', #no resuelve para este caso #MODIFICADO
     'Banknote', #no resuelve para este caso #MODIFICADO
    'Bank-marketing' # Sí
]
import pandas as pd
nombre_diccionario = 'datos_obct'
list_dataset = []
acc_media = []
desv_up = []
desv_low =[]
minimos = []
maximos = []
for dataset in lista_datasets:
    print(dataset)
    list_dataset.append(dataset)
    # x, y = loadData(dataset, binary = True)
    scor , media = cross_val_obct(dataset, 5)
    acc_media.append(media)
    # scor = scores.tolist()
    minimos.append(min(scor))
    maximos.append(max(scor))
    desv_low.append(abs(min(scor)- media))
    desv_up.append(abs(max(scor)-media))
diccionario = pd.DataFrame({'dataset': list_dataset,
                'acc_medio':acc_media,
                'desv_l':desv_low,
                'desv_u':desv_up,
                'minimos':minimos,
                'maximos':maximos})
nombre_archivo = f"C:/Users/pedro/Desktop/Semestre 2023-2/Proyecto de Datos 2/Códigos/uohpd2-main/{nombre_diccionario}.csv"

# Escribir el DataFrame en un archivo CSV
diccionario.to_csv(nombre_archivo, index=False)



################################################################################################################################

# x, y = loadData("Adult", binary=True)
# print(len(y))

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# d = 3
# n = 0  # Cantidad de nodos a fijar con preguntas que no son "buenas" pero rápidas
# # Cuando n = 1, entonces son 2 preguntas
# # n = 0 ==> 1 pregunta
# max_n = 4  # Valor máximo de n
# b, w, tiempo, gap = setMax4(x_train, y_train, d, 0, 4)
# y_pred_train = predict(b, w, x_train, d)
# y_pred_test = predict(b, w, x_test, d)
# accuracy_in = accuracy_score(y_train, y_pred_train)
# accuracy_out = accuracy_score(y_test, y_pred_test)

# print(accuracy_in)
# print(accuracy_out)