from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],28,28,1)
# Y_train = Y_train.reshape(Y_train.shape[0],28,28,1)

X_train = X_train / 255
# Y_train = Y_train / 255

Y_train = np_utils.to_categorical(Y_train)

model = Sequential()
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Flatten())
model.add(Dense(Y_train.shape[1], activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(X_train, Y_train, epochs=10, batch_size=32)