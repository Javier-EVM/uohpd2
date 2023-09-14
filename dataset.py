import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer

def loadData(dataname):
    # balance-scale
    if dataname == 'house-votes-84':
        x, y = loadHouseVotes84()
        return x, y
    
    if dataname == 'spambase':
        x, y = loadSpambase()
        return x, y



def oneHot(x):
    """
    one-hot encoding
    """
    x_enc = np.zeros((x.shape[0], 0))
    for j in range(x.shape[1]):
        lb = LabelBinarizer()
        lb.fit(np.unique(x[:,j]))
        x_enc = np.concatenate((x_enc, lb.transform(x[:,j])), axis=1)
    return x_enc

def loadSpambase():
    """
    load spambase dataset
    """
    """
    ejemplo de linea de datos:
    0,0.64,0.64,0,0.32,0,0,0,0,0,0,0.64,0,0,0,0.32,0,1.29,1.93,0,0.96,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.778,0,0,3.756,61,278,
    1  #clase
    """
    """
    57 atributos continuos, 1 clase 
    """

    df = pd.read_csv('./data/spambase/spambase.data', header=None, delimiter=',')
    #print(df)
    x, y = df.loc[:, 0:56], df[57] #df.loc[:, 0:56], : toma todas las filas, 0:56 toma las columnas de 0 a 56, contando 56
    return np.array(x), np.array(y)

def loadHouseVotes84():
    """
    load house-votes-84 dataset
    """
    """
    ejemplo de linea de datos:
    republican,n,y,n,y,y,y,n,n,n,y,?,y,y,y,n,y
    """
    """
    Son 16 atributos y 1 categoría
    """
    df = pd.read_csv('./data/house-votes-84/house-votes-84.data', header=None, delimiter=',')
    for i in range(1,17):
        df = df[df[i] != '?'] #df[i] != '?' es un filtro, revisa columna a columna por datos distintos de ?
        #que representan datos faltantes.
    
    #print(df)
    #df = df.apply(lambda x: pd.factorize(x)[0]) #no me gusto como funcionaba
    df = df.replace({"y": 1, "n": 0, "democrat":1, "republican":0})
    #print(df)
    # democrat,n,y,y,n,y,y,n,n,n,n,n,n,y,y,y,y 
    x, y = df[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]], df[0]
    return np.array(x), np.array(y)


x,y = loadData("house-votes-84")
x,y=loadData("spambase")

print(f"El tamaño del arreglo es: ",x.shape)
