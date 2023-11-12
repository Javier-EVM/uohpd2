import pandas as pd
import numpy as np
import sys
import math
import random
import csv
from gurobipy import *
import math
from arbol import tree

def predict(b, w, x_pred, d):
    #inicializar y de respuesta
    y_pred = len(x_pred)*[0]
    F = len(x_pred[0])
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
    
    for i,xi in enumerate(x_pred):
        n = 1
        for j in range(d-1):
             for f in range(F): #cambiar
                if n < 2**(d-1):
                    if 0.9 < b[n,f] < 1.1:
                        if 0.9< xi[f] < 1.1:
                            n = r(n)
                        else:
                            n = l(n)
                    else: #si no branchea en b[n,f] para todo f
                        #entonces se caracteriza en w[n,k] para algun k
                        for k in range(2): #IMPORTANTE
                            if 0.9 < w[n,k] < 1.1:
                                y_pred[i] = k
                else:
                    for k in range(2): #hay 2 caracteristicas
                            if 0.9 < w[n,k] < 1.1:
                                y_pred[i] = k
    return y_pred
             
        