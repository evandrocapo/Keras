import numpy as np  
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('dataset/IRIS.csv', header=None)

dataset = df.values

x = dataset[:,0:8].astype('float')
y = dataset[:, 8]

encoder = LabelEncoder()
encoded = encoder.fit_transform(y)

y = np_utils.to_categorical(encoded)

model = Sequential()

model.add(Dense(15, input_dim=8, activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])


print(x)