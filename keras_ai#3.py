import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

'''
    Dormiu Estudou -> Nota
        8       4       7
        10      4       8
        10      5       9.5
        5       7       4.5
        5       10      5
        10      6       10
'''


x = np.array([[0.8, 0.4],[0.10, 0.4],[0.10, 0.5],[0.5, 0.7],[0.5, 0.10]])
y = np.array([[0.7],[0.8],[0.95],[0.45],[0.5]])

model = Sequential() # modelo

model.add(Dense(5, input_dim=2)) # adicinou uma layer
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='sgd', loss='mse', metrics=['acc'])

model.fit(x, y, epochs=5000)

while True:

    dormiu = input('Dormiu: ')
    estudou = input('Estudou: ')

    lista = [float(dormiu) * 0.1, float(estudou) * 0.1]
    t = np.asmatrix(lista)
    result = model.predict(t)

    print(lista, ' previsto => ', result[0][0] * 10.0, ' como nota.')