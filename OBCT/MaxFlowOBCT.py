import pandas as pd
import numpy as np
import sys
import math
import random
import csv
from gurobipy import *
import math
from arbol import tree

def MFOBCT(x_train, y_train,d ,lambda_ ,rb = False, rw = False, Relaxed = False ):
    #numero de datos
    I_n = len(x_train)

    F_n = len(x_train[0])
    ###Arbol
    #d = 3
    s, t, N, L, NUL, s_NUL, NUL_t, A = tree(d)
    
    #funciones para retornar
    #ancestro
    def a(n):
        return A[n][0][0]
    #hijo izquierdo
    def l(n):
        return A[n][1][1]
    #hijo derecho
    def r(n):
        return A[n][2][1]


    #numero de clases (a predecir)
    K_n = 2

    #rango de los datos
    I = range(I_n)

    #rango de las features
    F = range(F_n)

    #rango de las clases
    K = range(K_n)

    #valor de alpha para controlar los branching
    #lda = 0
    model = Model()
    model.Params.LogToConsole = 0

    if Relaxed:
        b = model.addVars(N ,F , obj = 0, vtype = GRB.CONTINUOUS, name = "b")
        w = model.addVars(NUL, K , obj = 0, vtype = GRB.CONTINUOUS, name = "w")
        z = model.addVars(I, s_NUL, NUL_t , obj = 0, vtype = GRB.CONTINUOUS, name = "z")
    else:
        b = model.addVars(N ,F , obj = 0, vtype = GRB.BINARY, name = "b")
        w = model.addVars(NUL, K , obj = 0, vtype = GRB.BINARY, name = "w")
        z = model.addVars(I, s_NUL, NUL_t , obj = 0, vtype = GRB.BINARY, name = "z")
    #ocurre branch en la propiedad f en el nodo n?
    #b = model.addVars(N ,F , obj = 0, vtype = GRB.BINARY, name = "b")
    #la clase predecida pra un nodo n es k ¿Es L o NUL?
    #un data point esta correctamente clasificado si wnk = 1 con k = y
    #w = model.addVars(NUL, K , obj = 0, vtype = GRB.BINARY, name = "w")
    #z _a(n), n **i es igual a 1 ssi el dato i esta correctamente clasificado
    #y viaja por la arista (a(n), n)

    #z = model.addVars(I, s_NUL, NUL_t , obj = 0, vtype = GRB.BINARY, name = "z")
    #se debe restringir a aristas validas
    model.addConstrs(z[i,s,t] == 0 for i in I)


    if rb  != False:
        for i in rb:
            model.addConstr(b[i[0],i[1]] == 1, f"b{i}")

    
    if rw  != False:
        for i in rw:
            model.addConstr(w[i[0],i[1]] == 1, f"w{i}")
    


    model.setObjective((1-lambda_)*sum(z[i,n,t] for i in I for n in NUL) - lambda_*sum(b[n,f] for n in N for f in F), GRB.MAXIMIZE)


    model.addConstrs( sum(b[n,f] for f in F) + sum(w[n,k] for k in K) == 1 for n in N) #1.2

    model.addConstrs( sum(w[n,k] for k in K) == 1 for n in L) #1.3

    model.addConstrs( z[i,a(n),n] == z[i,n,l(n)] + z[i,n,r(n)] + z[i,n,t] for n in N for i in I) #1.4

    model.addConstrs( z[i,a(n),n] == z[i,n,t] for i in I for n in L) #1.5

    model.addConstrs( z[i,s,1] <= 1 for i in I) #1.6

    model.addConstrs( z[i,n,l(n)] <= sum(b[n,f] for f in F if (x_train[i][f] == 0)) for n in N for i in I ) #1.7

    model.addConstrs( z[i,n,r(n)] <= sum(b[n,f] for f in F if (x_train[i][f] == 1)) for n in N for i in I ) #1.8

    model.addConstrs( z[i,n,t] <= w[n,k] for i in I for n in NUL for k in K if (int(y_train[i]) == k)) #1.9

    model.setParam('TimeLimit', 40)
    model.optimize()
    
    obj = model.getObjective()


    gap = 0
    if Relaxed == False:
        gap = model.MIPGap
        print("Final MIP gap value: %f" % gap)
         
    
    correctamente_clasificados = sum(z[i,n,t].X for i in I for n in NUL)



    B = {}
    W = {}
    Z = {}
    #for name, val in zip(names, values):
        #if ("b" in name):
            #print(f"{name} = {val}")
        #if ("w" in name):
            #hola
            #print()

    for n in N: 
        for f in F:
            B[n,f] = b[n,f].X
            
            #if ((0.9 < b[n,f].X) and (b[n,f].X < 1.1)):
                #print(f'En el nodo {n}')
                #print(f'Se branchea en la feature {f}')
                #print(b[n,f].X)
                #b_count +=1
    #w_count = 0
    for n in NUL: 
        for k in K:
            W[n,k] = w[n,k].X
            #if ((0.9 < w[n,k].X) and (w[n,k].X < 1.1)):
                #print(f'En el nodo {n}')
                #print(f'la categoria escogida es {k}')
                #print(w[n,k].X)
                #w_count +=1
    return (model.Runtime, obj.getValue(), B, W, z, gap)
