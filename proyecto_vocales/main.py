import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Ruta a los archivos de datos
data_path = 'data/'

# Cargar los datos
x_train = np.load(data_path + 'x_train.npy')
y_train = np.load(data_path + 'y_train.npy')
x_test = np.load(data_path + 'x_test.npy')
y_test = np.load(data_path + 'y_test.npy')

# Normalizar los datos
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convertir las etiquetas a una codificación one-hot
y_train = to_categorical(y_train, 5)
y_test = to_categorical(y_test, 5)

# Crear el modelo
model = Sequential([
    Flatten(input_shape=x_train.shape[1:]), # Ajustar según el tamaño de entrada
    Dense(128, activation='relu'), # Primera capa intermedia
    Dense(64, activation='relu'), # Segunda capa intermedia
    Dense(5, activation='softmax') # Capa de salida con 5 neuronas (una para cada vocal)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")