# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:27:50 2018

@author: Chong
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import numpy as np

# Example from TensorFlow website
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


#
## Load mat data
#path = "C:/ChongDuan/6_DeepLearning_CMR-FT_Strain/Deep-MRI-Reconstruction-master/load_raw_T1_Map_data/sense_recon"
#mats = os.listdir(path)
#
#test = loadmat(os.path.join(path, mats[0]))['res']
#plt.imshow(np.abs(np.squeeze(test[:,:,1,0])))
#plt.show()
#
#row, col, t1w, sli = test.shape
#
#test = np.reshape(test, (row, col, -1))
#valid_mask = (np.abs(np.squeeze(test[int(row/2), int(col/2), :])) != 0)
#final_images = test[:,:,valid_mask]
#plt.imshow(np.abs(np.squeeze(final_images[:,:,50])))
#plt.show()
#
#for filename in mats:
#    if not filename.startswith('.'):
#        loadmat(os.path.join(path, filename))['res']