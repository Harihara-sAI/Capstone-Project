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
data_dir = 'C:/Capstone Image Data/'
img_size=180
batch_size=8
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_size, img_size),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_size, img_size),
  batch_size=batch_size)
#%%
class_names = train_ds.class_names
print(class_names)

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

#%%
num_classes = 6

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
  layers.Flatten(),
  layers.Dense(units=100, activation='relu'),
  layers.Dense(units=50, activation='relu'),
  layers.Dropout(0.25),
  layers.Dense(num_classes,activation='softmax')
])
     

#%%
optimizer = keras.optimizers.Adam(lr=1e-6)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#%%
epochs=100
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

#%%
tf.keras.models.save_model(model,'my_model.hdf5')

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

plt.savefig("results_final.png")