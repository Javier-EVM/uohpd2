from heuristica import setMax,setMax3,setMax4
from dataset import loadData
from sklearn.model_selection import train_test_split
from predict import predict
from sklearn.metrics import accuracy_score
from plotOBCT import plotOBCT
#b : Nodos brancheo
#w : Nodos clasificación

d = 3
alpha = 0
x,y = loadData("house-votes-84") 


x_train = x[0:100, :]
x_test = x[101:, :]
y_train = y[0:100]
y_test = y[101:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#b, w, tiempo = setMax(x_train, y_train, 3 , 0)
b, w, tiempo, gap = setMax4(x_train, y_train, d , alpha, 1)

y_pred_train = predict(b, w, x_train, d)
y_pred_test = predict(b, w, x_test, d)

accuracy_in = accuracy_score(y_train, y_pred_train)
accuracy_out = accuracy_score(y_test, y_pred_test)

print(accuracy_in)
print(accuracy_out)
print(tiempo)

#print(b)
#print(w)

with open(f"Salida-OBCT.txt", 'w') as f:
    f.write("B")
    for i in b:
        #if ( 0.9 < b[i] < 1.1 ):
        f.write(f"b({i}): {b[i]}")
        f.write('\n') 

    
    f.write('\n') 
    f.write("W")
    for j in w:
        #if (  0.9 < w[j] < 1.1 ):
        f.write(f"w({j}): {w[j]}")
        f.write('\n') 
    f.write('\n') 


plotOBCT(b,w,alpha,d)