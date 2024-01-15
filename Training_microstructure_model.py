# Imports 

import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D,MaxPool2D
from keras.utils import normalize, to_categorical
from keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split




plt.style.use('classic')

# Initializations

data_dir = "C:/Data for Capstone/data_dir"
test_dir="C:/Data for Capstone/test_dir"
SIZE = 200
batch_size = 8
num_classes=3

# Modifying images for easier working and labelling properly 


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(SIZE, SIZE),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(SIZE, SIZE),
  batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  subset=None,
  image_size=(SIZE,SIZE),
  batch_size=batch_size)


# Defining model

model = Sequential([
  layers.Rescaling(1./255,input_shape=(SIZE, SIZE, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.L2(0.01)),
  layers.Dense(num_classes, name="outputs"),
])


# Compiling the model

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.summary()
epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)



# Visualizing various parameters

plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range,acc, label='Training Accuracy')
plt.plot(epochs_range,val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.ylim(0,1)

plt.subplot(1, 2, 2)
plt.plot(epochs_range,loss, label='Training Loss')
plt.plot(epochs_range,val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.ylim(0,2)

string="SEM_microstructure_model_val_acc_v1.h5"

model.save(string)




from sklearn.metrics import  confusion_matrix, accuracy_score
predictions = model.predict(x=test_ds, verbose=0)

true_categories = tf.concat([y for x, y in test_ds], axis=0)
cm = confusion_matrix(y_true=true_categories, y_pred=np.argmax(predictions, axis=1))


cm_plot_labels = ['A','B','C']

print(cm)

print(accuracy_score(y_true=true_categories, y_pred=np.argmax(predictions, axis=1)))

