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
from generate_input import create_x_motion

dir_train = "/home/chongduan/Documents/Automap-MRI/Dataset"
data = loadmat(os.path.join(dir_train, 'Stone_all_crop_64'))['crop_data_resize']

img = np.abs(data[:,:,1,4,1])
plt.imshow(img, cmap='gray')
plt.show()

X = create_x_motion(img)

### Plot images
X_compl = X[:, :, :, 0] + X[:, :, :, 1] * 1j

im_artif0 = np.fft.ifft2(X_compl[0, :, :])

img_artif_M0 = np.abs(im_artif0)

plt.figure()
plt.subplot(131), plt.imshow(np.abs(X_compl[0,:,:]), cmap='gray')
plt.title('k-space'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_artif_M0, cmap='gray')
plt.title('ifft'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img, cmap='gray')
plt.title('groundTruth'), plt.xticks([]), plt.yticks([])