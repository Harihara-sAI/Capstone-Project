#%% Imports 

import os
# import cv2
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
from keras.applications import ResNet50V2


plt.style.use('classic')

#%% Initializations

train_dir = "C:/Minimal Cap Data/train_dir"
val_dir = "C:/Minimal Cap Data/val_dir"
test_dir="C:/Minimal Cap Data/test_dir"
SIZE = 100
batch_size = 4
num_classes=3

#%% Modifying images for easier working and labelling properly 


train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  subset=None,
  image_size=(SIZE, SIZE),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  val_dir,
  subset=None,
  image_size=(SIZE, SIZE),
  batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  subset=None,
  image_size=(SIZE,SIZE),
  batch_size=batch_size)


#%% Defining model

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

#model = Sequential([
  #layers.Rescaling(1./255,input_shape=(SIZE, SIZE, 3)),
  #data_augmentation,
  #layers.Conv2D(16, 3, padding='same', activation='relu'),
  #layers.MaxPooling2D(),
  #layers.Conv2D(32, 3, padding='same', activation='relu'),
  #layers.MaxPooling2D(),
  #layers.Conv2D(64, 3, padding='same', activation='relu'),
  #layers.MaxPooling2D(),
  #layers.Conv2D(128, 3, padding='same', activation='relu'),
  #layers.MaxPooling2D(),
  #layers.Dropout(0.2),
  #layers.Flatten(),
  #layers.Dense(128, activation='relu'),
  #layers.Dense(num_classes, name="outputs",activity_regularizer=tf.keras.regularizers.L2(0.01)),
#])

#%%

model=Sequential()
pre_model= ResNet50V2(input_shape=(SIZE,SIZE,3),pooling='max',classes=3,weights=None)
for each_layer in pre_model.layers:

        each_layer.trainable=True

model.add(pre_model)
model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(3, activation='relu', name="outputs", activity_regularizer=tf.keras.regularizers.L2(0.01))
)
#%% Compiling the model

optimizer = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.summary()
epochs=200
checkpoint_filepath = 'C:/Users/hahas/OneDrive/GitHub/Capstone-Project/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=epochs,
  callbacks=[model_checkpoint_callback]
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)



#%% Visualizing various parameters

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

plt.savefig("results.png")

string="SEM_microstructure_model_val_acc_v2.h5"

model.save(string)

#%% Testing with actual data

from sklearn.metrics import  confusion_matrix, accuracy_score
predictions = model.predict(x=val_ds, verbose=0)

true_categories = tf.concat([y for x, y in val_ds], axis=0)
cm = confusion_matrix(y_true=true_categories, y_pred=np.argmax(predictions, axis=1))


cm_plot_labels = ['CPJ','HR','P92']

print(cm)

print(accuracy_score(y_true=true_categories, y_pred=np.argmax(predictions, axis=1)))


# %%
