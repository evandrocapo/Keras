from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

num_pix = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], num_pix).astype('float')
X_test = X_test.reshape(X_test.shape[0], num_pix).astype('float')

x_train = X_train / 255
x_test = X_test / 255

y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)

classes = y_test.shape[1]

model = Sequential()

model.add(Dense(120, input_dim=num_pix, activation='relu'))
model.add(Dense(classes, activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, epochs=12, batch_size=100)

# plt.imshow(X_train[0])

# plt.show()