from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import numpy as np

# Criação do modelo complicado

    # model = Sequential([
    #     Dense(32, input_shape=(784,)),
    #     Activation('relu'),
    #     Dense(10),
    #     Activation('softmax'),
    # ])


# Criação do modelo simples

model = Sequential()
model.add(Dense(32, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))


# Compilador

    # Para classificação binaria

    # model.compile(optimizer='rmsprop',
    #               loss='binary_crossentropy',
                #   metrics=['accuracy'])

    # Para classificação mean squared

    # model.compile(optimizer='remsprop',
                #   loss='mse')



# Para multi classes
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Generate Data

# Binary
# data = np.random.random((1000,100))
# labels = np.random.randint(2, size=(1000,1))

# Multi-classes

data = np.random.random((1000,100))
labels = np.random.randint(10, size=(1000,1))

# Converter labels para categorias 'one-hot encoding'

one_hot_labels = to_categorical(labels, num_classes=10)

#Training

# Binary
# model.fit(data,labels, epochs=10, batch_size = 32)

#multi classes

model.fit(data, one_hot_labels, epochs=10, batch_size=32)