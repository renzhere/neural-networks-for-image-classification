import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

from glob import glob

# from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from google.colab import drive
drive.mount('/content/gdrive')


def loadDataH5():
  with h5py.File('/content/gdrive/My Drive/Colab Notebooks/ASSIGNMENTS/Assignment2/data1.h5','r') as hf:
    trainX = np.array(hf.get('trainX'))
    trainY = np.array(hf.get('trainY'))
    valX = np.array(hf.get('valX'))
    valY = np.array(hf.get('valY'))
    print (trainX.shape,trainY.shape)
    print (valX.shape,valY.shape)
    
    
  return trainX, trainY, valX, valY


def CNNmodel(width, height, depth):
  
  model = Sequential()
  input_shape = (width, height, depth)

  #first convolutional and pooling layers
  model.add(Conv2D(16, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  
  #second convolutional and pooling layers
  model.add(Conv2D(32, (3, 3), padding = "same", activation = 'relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(17, activation='softmax'))
  
  return model

  
def main():
  
  trainX, trainY, valX, valY = loadDataH5()
  
  # resize all the images into the height and width required
  width, height, depth = 128, 128, 3
  
  print(trainX.shape)

  NUM_EPOCHS = 20
  opt = tf.keras.optimizers.SGD(lr=0.01)
  
#   trainX = trainX.astype("float")
#   valX = valX.astype("float")

  model = CNNmodel(width, height, depth)

  model.compile(loss='sparse_categorical_crossentropy', 
                optimizer = opt, metrics=['accuracy'])
  model.summary()

  H = model.fit(trainX, trainY, validation_data = (valX, valY), 
                          batch_size=32,epochs = NUM_EPOCHS)
  
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H.history["acc"], label="train_acc")
  plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  plt.show()
  
  
  
  
main()  
