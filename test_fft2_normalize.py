#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:15:50 2018

@author: chongduan
"""

### Without normalization
X_dev_compl = X_dev[:, :, :, 0] + X_dev[:, :, :, 1] * 1j

#iFFT
X_iFFT0 = np.fft.ifft2(X_dev_compl[im1, :, :])
X_iFFT1 = np.fft.ifft2(X_dev_compl[im2, :, :])
X_iFFT2 = np.fft.ifft2(X_dev_compl[im3, :, :])
X_iFFT3 = np.fft.ifft2(X_dev_compl[im4, :, :])

# Magnitude of complex image
X_iFFT_M1 = np.sqrt(np.power(X_iFFT0.real, 2)
                    + np.power(X_iFFT0.imag, 2))
X_iFFT_M2 = np.sqrt(np.power(X_iFFT1.real, 2)
                    + np.power(X_iFFT1.imag, 2))
X_iFFT_M3 = np.sqrt(np.power(X_iFFT2.real, 2)
                    + np.power(X_iFFT2.imag, 2))
X_iFFT_M4 = np.sqrt(np.power(X_iFFT3.real, 2)
                    + np.power(X_iFFT3.imag, 2))

# SHOW
# Show Y - input images
plt.subplot(241), plt.imshow(Y_dev[im1, :, :], cmap='gray')
plt.title('Y_dev1'), plt.xticks([]), plt.yticks([])
plt.subplot(242), plt.imshow(Y_dev[im2, :, :], cmap='gray')
plt.title('Y_dev2'), plt.xticks([]), plt.yticks([])
plt.subplot(243), plt.imshow(Y_dev[im3, :, :], cmap='gray')
plt.title('Y_dev3'), plt.xticks([]), plt.yticks([])
plt.subplot(244), plt.imshow(Y_dev[im4, :, :], cmap='gray')
plt.title('Y_dev4'), plt.xticks([]), plt.yticks([])

# Show images reconstructed using iFFT
plt.subplot(245), plt.imshow(X_iFFT_M1, cmap='gray')
plt.title('X_iFFT1'), plt.xticks([]), plt.yticks([])
plt.subplot(246), plt.imshow(X_iFFT_M2, cmap='gray')
plt.title('X_iFFT2'), plt.xticks([]), plt.yticks([])
plt.subplot(247), plt.imshow(X_iFFT_M3, cmap='gray')
plt.title('X_iFFT3'), plt.xticks([]), plt.yticks([])
plt.subplot(248), plt.imshow(X_iFFT_M4, cmap='gray')
plt.title('X_iFFT4'), plt.xticks([]), plt.yticks([])



### With normalization
X_dev_norm = X_dev / np.max(X_dev)
X_dev_compl = X_dev_norm[:, :, :, 0] + X_dev_norm[:, :, :, 1] * 1j

#iFFT
X_iFFT0 = np.fft.ifft2(X_dev_compl[im1, :, :])
X_iFFT1 = np.fft.ifft2(X_dev_compl[im2, :, :])
X_iFFT2 = np.fft.ifft2(X_dev_compl[im3, :, :])
X_iFFT3 = np.fft.ifft2(X_dev_compl[im4, :, :])

# Magnitude of complex image
X_iFFT_M1 = np.sqrt(np.power(X_iFFT0.real, 2)
                    + np.power(X_iFFT0.imag, 2))
X_iFFT_M2 = np.sqrt(np.power(X_iFFT1.real, 2)
                    + np.power(X_iFFT1.imag, 2))
X_iFFT_M3 = np.sqrt(np.power(X_iFFT2.real, 2)
                    + np.power(X_iFFT2.imag, 2))
X_iFFT_M4 = np.sqrt(np.power(X_iFFT3.real, 2)
                    + np.power(X_iFFT3.imag, 2))

# SHOW
# Show Y - input images
plt.subplot(241), plt.imshow(Y_dev[im1, :, :], cmap='gray')
plt.title('Y_dev1'), plt.xticks([]), plt.yticks([])
plt.subplot(242), plt.imshow(Y_dev[im2, :, :], cmap='gray')
plt.title('Y_dev2'), plt.xticks([]), plt.yticks([])
plt.subplot(243), plt.imshow(Y_dev[im3, :, :], cmap='gray')
plt.title('Y_dev3'), plt.xticks([]), plt.yticks([])
plt.subplot(244), plt.imshow(Y_dev[im4, :, :], cmap='gray')
plt.title('Y_dev4'), plt.xticks([]), plt.yticks([])

# Show images reconstructed using iFFT
plt.subplot(245), plt.imshow(X_iFFT_M1, cmap='gray')
plt.title('X_iFFT1'), plt.xticks([]), plt.yticks([])
plt.subplot(246), plt.imshow(X_iFFT_M2, cmap='gray')
plt.title('X_iFFT2'), plt.xticks([]), plt.yticks([])
plt.subplot(247), plt.imshow(X_iFFT_M3, cmap='gray')
plt.title('X_iFFT3'), plt.xticks([]), plt.yticks([])
plt.subplot(248), plt.imshow(X_iFFT_M4, cmap='gray')
plt.title('X_iFFT4'), plt.xticks([]), plt.yticks([])
