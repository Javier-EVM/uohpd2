from classifier import useClassifier 
from dataset import loadData 

x,y = loadData("Adult")

a_in, a_out, clf = useClassifier("Arbol",x,y,size = 0.2)


print(a_in)
print(a_out)
print(clf)

