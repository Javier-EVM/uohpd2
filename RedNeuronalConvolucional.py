import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset import loadData
import numpy as np

x,y = loadData('house-votes-84')
#x,y = loadData('spambase')
x,y = loadData("Credit-approval")
x,y = loadData("Ionosphere")
#entradas importantes
n_entrada= x.shape[1]
epoch = 50
#tipo de optimizador
#

print(n_entrada)

# Se divide el conjunto de entrenamiento en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Se normalizan las caracteristicas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Se crea el modelo de red neuronal
#con una capa oculta basta
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_entrada,)),  
    tf.keras.layers.Dense(2*n_entrada, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

#Se compila el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.BinaryAccuracy(name = 'accuracy'),
                       tf.keras.metrics.Precision(name = 'precision'),
                       tf.keras.metrics.Recall(name = 'recall')])

#Entrenamiento

history = model.fit(X_train, y_train, epochs=epoch)


import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["figure.figsize"] = (8,6)
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False

plt.plot(np.arange(1,epoch+1),history.history["loss"],label = "loss")
plt.plot(np.arange(1,epoch+1),history.history["accuracy"],label = "Accuracy")
plt.plot(np.arange(1,epoch+1),history.history["precision"],label = "Precision")
plt.plot(np.arange(1,epoch+1),history.history["recall"],label = "Recall")
plt.title("Evaluation metrics", size = "20")
plt.xlabel("Epoch", size = "14")
plt.legend()
plt.show()

print(history.history["accuracy"][-1])
#loss desviacion entre lo predecido y las categorias reales