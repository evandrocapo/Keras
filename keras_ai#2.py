import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

x = np.array([[0.1],[0.2],[0.3]])
y = np.array([[0.2],[0.4],[0.6]])

model = Sequential() # modelo

model.add(Dense(3, input_dim=1)) # adicinou uma layer
model.add(Dense(1))

model.compile(optimizer='sgd', loss='mse', metrics=['acc'])

model.fit(x, y, epochs=8000)

while True:

    i = input('Digite um numero: ')
    t = float(i)
    t = np.asmatrix(t)
    result = model.predict(t)

    print(i, ' previsto => ', result[0])