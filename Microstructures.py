# Imports 

import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
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

data_dir = "C:/Users/hahas/Downloads/Capstone Microstructure Data/"
SIZE = 200
batch_size = 15
