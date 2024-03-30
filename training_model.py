#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.style.use('classic')
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import normalize, to_categorical
import os
import cv2
from keras import layers
from PIL import Image
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout
from keras.models import Sequential
import tensorflow as tf

#%%
image_directory = 'C:/20BME0147/Capstone Data/'
SIZE = 200
dataset = []
label = []
dead_mild_steel = os.listdir(image_directory + '1 - Dead Mild Steel/')
for i, image_name in enumerate(dead_mild_steel):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + '1 - Dead Mild Steel/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append("Dead Mild Steel")

low_carbon_steel = os.listdir(image_directory + '2 - Low Carbon Steel/')
for i, image_name in enumerate(low_carbon_steel):
      if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + '2 - Low Carbon Steel/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append("Low Carbon Steel")

hardened_steel = os.listdir(image_directory + '7 - Hardened Steel/')
for i, image_name in enumerate(hardened_steel):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + '7 - Hardened Steel/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append("Hardened Steel")

tempered_steel = os.listdir(image_directory + '8 - Tempered Steel/')
for i, image_name in enumerate(hardened_steel):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + '8 - Tempered Steel/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append("Tempered Steel")

tool_steel = os.listdir(image_directory + '10 - Tool Steel/')
for i, image_name in enumerate(hardened_steel):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + '10 - Tool Steel/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append("Tool Steel")


#%%
dataset = np.array(dataset)
label = np.array(label)

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)

#%%
num_classes = 5

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(SIZE,SIZE, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#%%
model.summary()
epochs=100
history = model.fit(
  X_train,y_train,
  validation_data=(X_test,y_test),
  epochs=epochs
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

#%%
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range,acc, label='Training Accuracy')
plt.plot(epochs_range,val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range,loss, label='Training Loss')
plt.plot(epochs_range,val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()