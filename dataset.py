import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

def loadData(dataname):
    # balance-scale
    if dataname == 'house-votes-84':
        x, y = loadHouseVotes84()
        return x, y
    
    if dataname == 'spambase':
        x, y = loadSpambase()
        return x, y
    
    if dataname == "Monks1":
        x,y = loadMonks1()
        return x, y
    
    if dataname == "Monks2":
        x,y = loadMonks1()
        return x, y
    
    if dataname == "Monks3":
        x,y = loadMonks1()
        return x, y
    
    if dataname == "Credit-approval":
        x,y = loadCreditapproval()
        return x, y
    
    if dataname == "Ionosphere":
        x,y = loadIonosphere()
        return x, y
    
    if dataname == "Adult":
        x,y = loadAdult()
        return x, y
    




def oneHot(x):
    """
    one-hot encoding
    """
    x_enc = np.zeros((x.shape[0], 0)) #inicia matriz de numero de filas de x con 0 columnas
    for j in range(x.shape[1]): #recorre las columnas
        lb = LabelBinarizer() #instancia labelBinarizer
        lb.fit(np.unique(x[:,j])) #lo fitea con todas las filas de la columna j
        x_enc = np.concatenate((x_enc, lb.transform(x[:,j])), axis=1) #concatena x_enc con la binarizacion de la columna j, en el axis 1 == columna
    return x_enc

def loadAdult():
    df = pd.read_csv("./data/adult/adult.csv")
    for col in df.columns:
        df = df[df[col] != ' ?']
    #for col in df.columns:
    #    if " ?" in df[col].unique():
    #        df = df[df[col] != " ?"]
    #    else:
    #        continue
    dfx = df.iloc[:, :-1]  # Todas las columnas excepto la última como características
    dfy = df.iloc[:, -1]   # Última columna como etiquetas
    
    col_num = dfx.select_dtypes(include=['int64', 'float64']).columns
    col_cat = dfx.select_dtypes(include=['object','category']).columns
    '''
    Se estandarizan las variables numéricas
    '''
    sc = StandardScaler()
    
    dfx[col_num] = sc.fit_transform(dfx[col_num])
    '''
    Se codifican las columnas categóricas usando one-hot
    '''
    dfx = pd.get_dummies(dfx, columns=col_cat, prefix=col_cat)
    #print(f'Datos atributos:\n\n{dfx}')
    '''
    Se transforma dfx en array
    '''
    #print(dfx)
    dfx = np.array(dfx)
    #print(dfx)
    '''
    Se codifica la última columna a 0s y 1s
    '''
    dfy = LabelEncoder().fit_transform(dfy.values)
    return dfx,dfy

def loadIonosphere():
    """
    Ejemplo de linea de datos
    1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,
    -0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,
    -0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300,
    g
    
    """
    df = pd.read_csv('./data/Ionosphere/ionosphere.data', header=None, delimiter=',')
    #Se normaliza
    scaler = StandardScaler()
    data = df.loc[:, 0:33]
    scaler.fit(data)
    df.loc[:, 0:33] = scaler.fit_transform(data)
    #La categoria se pasa a binario
    df_et = df[34]
    y = df_et.replace({"g": 1, "b": 0})
    x, y = df.loc[:, 0:33], y #df.loc[:, 0:56], : toma todas las filas, 0:56 toma las columnas de 0 a 56, contando 56
    return np.array(x), np.array(y)

def loadCreditapproval():
    """
    Ejemplo de linea de datos
    b,34.83,4,u,g,d,bb,12.5,t,f,0,t,g,?,0,-
    
    """
    df = pd.read_csv('./data/credit-approval/crx.data', header=None, delimiter=',')
    #Elimina datos nulos
    for i in range(0,16):
        df = df[df[i] != '?']
    #Se separa etiqueta
    df_et = df[15]
    #Se separan los datos categoricos y continuos
    df_cont = df.iloc[:,[1,2,7,10,13,14]]
    df_cat = df.iloc[:,[0,3,4,5,6,8,9,11,12]]
    #Se aplica normalizacion
    scaler = StandardScaler()
    scaler.fit(df_cont)
    df_cont = scaler.fit_transform(df_cont)
    df_cont = np.array(df_cont)
    #Se pasa a onehot
    df_cat = oneHot(np.array(df_cat))
    x = np.concatenate((df_cont,df_cat) , axis=1)
    
    #La cateogria se pasa a binario
    y = df_et.replace({"+": 1, "-": 0})
    return x, np.array(y)

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
    scaler = StandardScaler()
    data = df.loc[:, 0:56]
    scaler.fit(data)
    df.loc[:, 0:56] = scaler.fit_transform(data)
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


    
def loadMonks1():
    """
    ejemplo de linea de datos:
     0 1 2 2 2 4 2 data_88
    
    Notar que df_train[0] = NaN,
    ,al leer toma el espacio como NaN
    """
    df_train = pd.read_csv('./data/monks/monks-1.train', header=None, delimiter=' ')
    df_test = pd.read_csv('./data/monks/monks-1.test', header=None, delimiter=' ')
    x_train, y_train = df_train[[2,3,4,5,6,7]], df_train[1]
    x_test, y_test = df_test[[2,3,4,5,6,7]], df_test[1]
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    x = oneHot(x)
    return x, y

def loadMonks2():
    """
    ejemplo de linea de datos:
     0 1 2 2 2 4 2 data_88
    
    Notar que df_train[0] = NaN,
    ,al leer toma el espacio como NaN
    """
    df_train = pd.read_csv('./data/monks/monks-2.train', header=None, delimiter=' ')
    df_test = pd.read_csv('./data/monks/monks-2.test', header=None, delimiter=' ')
    x_train, y_train = df_train[[2,3,4,5,6,7]], df_train[1]
    x_test, y_test = df_test[[2,3,4,5,6,7]], df_test[1]
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    x = oneHot(x)
    return x, y

def loadMonks1():
    """
    ejemplo de linea de datos:
     0 1 2 2 2 4 2 data_88
    
    Notar que df_train[0] = NaN,
    ,al leer toma el espacio como NaN
    """
    df_train = pd.read_csv('./data/monks/monks-3.train', header=None, delimiter=' ')
    df_test = pd.read_csv('./data/monks/monks-3.test', header=None, delimiter=' ')
    x_train, y_train = df_train[[2,3,4,5,6,7]], df_train[1]
    x_test, y_test = df_test[[2,3,4,5,6,7]], df_test[1]
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    x = oneHot(x)
    return x, y


#x,y = loadData("house-votes-84")
#x,y = loadData("spambase")
#x,y = loadData("Monks1")

#print(np.zeros((x.shape[0], 0)))
#print(f"El tamaño del arreglo es: ",x.shape)

#x,y = loadData("Credit-approval")
#x,y = loadData("Ionosphere")
#x,y = loadData("Adult")
#print(x)
#print(y)
#print(len(y))
#print(x.shape)