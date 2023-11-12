import pandas as pd
import numpy as np
import sys
import math
import random
import csv
from gurobipy import *
import math
from arbol import tree
from MaxFlowOBCT import MFOBCT
from sklearn.model_selection import train_test_split # Import train_test_split function
from collections import Counter

def takeMax(b, w):
    #for i, (nb, value) in enumerate(b.items()):
    max_valueb = max(b, key=b.get)
    max_valuew = max(w, key=w.get)
    #hay que agregar si w esta en el mismo piso o inferior, w max es valido
    #caso contrario recalcular
    #o calcular bmax y siguiente eliminar todo lo que sea de piso superior de w
    #recien ahi calcular max.
    bl = []
    wl = []
    bl.append(list(max_valueb))
    wl.append(list(max_valuew))

    return(bl,wl)



def takeMax2(b, w, d):  #por piso, revisar el maximo
    #for i, (nb, value) in enumerate(b.items()):
    max_each_depth = {}
    for i in range(d):
        aux_b = {}
        for n_f in b.keys():
            if 2**(i) <= n_f[0] <= (2**(i+1) -1):
                aux_b[n_f] = b[n_f]

        if aux_b != {}:
            max_key_d = max(aux_b, key=aux_b.get)
            max_each_depth[max_key_d] = b[max_key_d]

    max_index_b = max(max_each_depth, key=max_each_depth.get)   


    aux_w = {}
    for n_k in w.keys():
        if n_k[0] > max_index_b[0]:
            aux_w[n_k] = w[n_k]

    max_index_w = max(aux_w, key=aux_w.get)   
    

    #max_valueb = max(b, key=b.get)
    #max_valuew = max(w, key=w.get)
    #hay que agregar si w esta en el mismo piso o inferior, w max es valido
    #caso contrario recalcular
    #o calcular bmax y siguiente eliminar todo lo que sea de piso superior de w
    #recien ahi calcular max.
    bl = []
    wl = []
    bl.append(list(max_index_b))
    wl.append(list(max_index_w))

    return(bl,wl)


def takeMax_cut1(b, w, d):  #por piso, revisar el maximo
    
    max_each_depth = {} #lista vacia, se guardara el maximo en cada piso
    for i in range(d):
        aux_b = {}
        for n_f in b.keys(): #se recorren las variables de decisión
            if (2**(i) <= n_f[0] <= (2**(i+1) -1)) and (b[n_f] != 1):
                aux_b[n_f] = b[n_f] 

        if aux_b != {}: 
            max_key_d = max(aux_b, key=aux_b.get) #tomo el maximo del piso
            max_each_depth[max_key_d] = b[max_key_d] #lo guardo en la lista inicial

    max_index_b = max(max_each_depth, key=max_each_depth.get) #de la lista de maximos totales, tomo el maximo  
    bl = []
    bl.append(list(max_index_b))

    return(bl)


def takeMax3(b,w,d): #no se usa
    bl = []
    max_each_depth = {}
    for i in range(d):
        aux_b = {}
        for n_f in b.keys():
            if 2**(i) <= n_f[0] <= (2**(i+1) -1):
                aux_b[n_f] = b[n_f]

        if aux_b != {}:
            max_key_d = max(aux_b, key=aux_b.get)
            max_each_depth[max_key_d] = b[max_key_d]
    
    #Problema aqui, al tomar todos los maximos es posible tomar (1,7) y (1,7) por ejemplo, esto claramente es infactible.
    #max_value = max(b.values())
    #bl = [k for k,v in b.items() if v == max_value]  #recorre par llave, valor y appenda la llave a la lista, si el valor es maximo
    #list_n = sorted(bl, key=lambda element: element[0], reverse=True) #lista de tuplas n,f ordenadas por n descendiente
    #last_n= list_n[0][0]
    
    bl = list(max_each_depth.keys())
    
    s, t, N, L, NUL, s_NUL, NUL_t, A = tree(d)
    

    def a(n):
        return A[n][0][0]
    #hijo izquierdo
    def l(n):
        return A[n][1][1]
    #hijo derecho
    def r(n):
        return A[n][2][1]
    
    wl = []
    for k,v  in w.items():
        if k[0] in L: #n esta en Hojas
            if v >= 0.9:
                wl.append(k)

    return bl,wl

def randomSolution(x_enc,y,d,n,size):
    for i in range(n):
        if i == 0:
            x_train, x_test, y_train, y_test = train_test_split(x_enc, y, test_size = size) #se entrena con test y prueba con train
            t, of , b , w , z = MFOBCT(x_test, y_test ,d ,0, False, False, True)
            B = b.copy()
            W = w.copy()
        else:
            x_train, x_test, y_train, y_test = train_test_split(x_enc, y, test_size = size)
            t, of , b , w , z = MFOBCT(x_test, y_test ,d ,0, False, False, True)
            B = dict(Counter(b)+ Counter(B))
            W = dict(Counter(w)+ Counter(W))
    

    s, t, N, L, NUL, s_NUL, NUL_t, A = tree(d)
    
    def a(n):
        return A[n][0][0]
    #hijo izquierdo
    def l(n):
        return A[n][1][1]
    #hijo derecho
    def r(n):
        return A[n][2][1]
    
    B_fac = {}
    for n in N:
        aux_b = {k: B[k] for k in set(B) if k[0] == n} #creo un diccionario con solo las llaves n',k para todo k y un n' especifico
        if aux_b != {}:
            max_key_d = max(aux_b, key=aux_b.get)
            B_fac[max_key_d] = B[max_key_d]

    W_fac = {}
    for n in L:
        aux_w = {k: W[k] for k in set(W) if k[0] == n} #creo un diccionario con solo las llaves n',k para todo k y un n' especifico
        if aux_w != {}:
            max_key_d = max(aux_w, key=aux_w.get)
            W_fac[max_key_d] = W[max_key_d]


    b_final = {}
    for k in set(B_fac):
        b_final[k[0],k[1]] = 1
    
    w_final = {}
    for k in set(W_fac):
        w_final[k[0],k[1]] = 1

    #return B,B_fac,W,W_fac
    #return B_fac,b_final,W_fac,w_final

    b_zeros = {}
    w_zeros = {}
    for n in N:
        for f in range(38):
            b_zeros[n,f] = 0
    
    for n in NUL:
        w_zeros[n,0] = 0
        w_zeros[n,1] = 0
    be = {k: b_zeros.get(k, 0) + b_final.get(k, 0) for k in set(b_zeros) | set(b_final)}
    we = {k: w_zeros.get(k, 0) + w_final.get(k, 0) for k in set(w_zeros) | set(w_final)}
    #b_final = dict(Counter(b_final)+ Counter(b_zeros))
    #w_final = dict(Counter(w_final)+ Counter(w_zeros))



    return be,we




