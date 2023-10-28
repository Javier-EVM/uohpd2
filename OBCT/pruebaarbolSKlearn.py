from heuristica import setMax,setMax3,setMax4
from dataset import loadData
from sklearn.model_selection import train_test_split
from predict import predict
from sklearn.metrics import accuracy_score
from classifierTree import classifierTree
#b : Nodos brancheo
#w : Nodos clasificación


x,y = loadData("Monks1") 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



accuracy_in, accuracy_out, params, clf = classifierTree(x,y,0.2,None)


print(accuracy_in)
print(accuracy_out)

from sklearn.tree import export_graphviz
import graphviz

# Exportar el árbol a un archivo DOT
export_graphviz(clf, out_file="tree.dot", feature_names=data.feature_names, class_names=data.target_names, filled=True, rounded=True)


# Crear una representación gráfica del árbol desde el archivo DOT
dot_data = graphviz.Source(open("tree.dot", "r").read())
dot_data.format = "png"  # También puedes usar otros formatos como "pdf" o "svg"
dot_data.render("tree")