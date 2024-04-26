#%%

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import BatchNormalization, Dropout
#%%
num_classes = 6
img_size=180
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_size, img_size, 3)),
  layers.Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'),
  layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
  layers.Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'),
  layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
  #layers.BatchNormalization(),
  layers.Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'),
  layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
  #layers.BatchNormalization(),
  #layers.Conv2D(100, (2, 2), activation='relu', strides=(2, 2), padding='same'),
  #layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
  #layers.BatchNormalization(),
  layers.Flatten(),
  layers.Dense(units=100, activation='relu'),
  layers.Dense(units=50, activation='relu'),
  layers.Dropout(0.25),
  layers.Dense(num_classes,activation='softmax')
])
#%%

from neuralplot import ModelPlot
modelplot = ModelPlot(model=model, grid=True, connection=True, linewidth=0.1)
modelplot.show()