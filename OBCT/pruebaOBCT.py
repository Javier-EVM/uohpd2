from heuristica import setMax,setMax3,setMax4
from dataset import loadData
from sklearn.model_selection import train_test_split
from predict import predict
from sklearn.metrics import accuracy_score
#b : Nodos brancheo
#w : Nodos clasificación


x,y = loadData("Monks1") 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#b, w, tiempo = setMax(x_train, y_train, 3 , 0)
b, w, tiempo, gap = setMax4(x_train, y_train, 5 , 0, 2)

y_pred_train = predict(b, w, x_train, 5)
y_pred_test = predict(b, w, x_test, 5)

accuracy_in = accuracy_score(y_train, y_pred_train)
accuracy_out = accuracy_score(y_test, y_pred_test)

print(accuracy_in)
print(accuracy_out)
print(tiempo)
