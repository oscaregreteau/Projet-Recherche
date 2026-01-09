import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

model=keras.Sequential([
    keras.layers.Dense(1,input_shape=[1])
])
#keras.layers.Dense(1024, activation='relu'),

model.compile(optimizer='sgd',
              loss='mean_squared_error')

x=np.array([1,2,3,4,5],dtype=float)
y=np.array([1,3,5,7,9],dtype=float)
model.fit(x,y,epochs=500)
print(model.predict(np.array([18],dtype=float)))