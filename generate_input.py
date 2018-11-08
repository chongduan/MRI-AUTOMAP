import numpy as np
import cv2
import os

# Chong Duan
from scipy.io import loadmat
from matplotlib import pyplot as plt


def load_STONE_data(folder, n_cases, normalize=False, imrotate=False, motion=False):
    """
    """
    temp = loadmat(os.path.join(folder, 'Stone_all_crop_64'))['crop_data_resize']
    row, col, t1w, sli, n = temp.shape
    temp = np.reshape(temp[:,:,:,:,n_cases[0]:n_cases[1]], (row, col, -1))
    bigy = np.transpose(temp, (2,0,1))
    
    
    # normalize
    if normalize:
        bigy = np.abs(bigy)
        temp_bigy = np.reshape(bigy, (55, n_cases[1] - n_cases[0], row, col))
        for i in range(n_cases[1] - n_cases[0]):
            temp_bigy[:,i,:,:] = (temp_bigy[:,i,:,:] - np.min(temp_bigy[:,i,:,:])) / (np.max(temp_bigy[:,i,:,:]) -np.min(temp_bigy[:,i,:,:]))
        
        bigy = np.reshape(temp_bigy, (-1, row, col))
    
    # convert to k-space
    imgs, row, col = bigy.shape
    if imrotate:
        bigx_rot_all = []
        bigy_rot_all = []
        # cv2 rotate does not work on complex data
        bigy = np.abs(bigy)
        for i in range(imgs):
            temp_image = np.squeeze(bigy[i,:,:])
            bigy_rot_all.append(temp_image) 
            bigx_rot_all.append(create_x(temp_image, motion))
            for angle in [45, 90, 135, 180, 225, 270]:
                bigy_rot = im_rotate(temp_image, angle)
                bigy_rot_all.append(bigy_rot)
                bigx_rot_all.append(create_x(bigy_rot, motion))
        
        
        bigx_rot_all = np.asarray(np.squeeze(bigx_rot_all))
        bigy_rot_all = np.asarray(np.abs(bigy_rot_all))

        # Pad bigy_rot_all if motion is included
        if motion:
            temp_bigy = bigy_rot_all.copy()
            bigy_rot_all = np.zeros((temp_bigy.shape[0], 80, 80))
            bigy_rot_all[:,8:72, 8:72] = temp_bigy
        
        return  bigx_rot_all, bigy_rot_all
                            
    else:
        bigx = []
        bigy = np.abs(bigy)
        for i in range(imgs):
            bigx.append(create_x(np.squeeze(bigy[i,:,:]), motion))

        bigx = np.asarray(np.squeeze(bigx))
        
        # Pad bigy if motion is included
        if motion:
            temp_bigy = bigy.copy()
            bigy = np.zeros((temp_bigy.shape[0], 80, 80))
            bigy[:,8:72, 8:72] = temp_bigy        
        
        return bigx, bigy
    
    
def load_images_from_folder(folder, n_cases, normalize=False, imrotate=False):
    """ Loads n_im images from the folder and puts them in an array bigy of
    size (n_im, im_size1, im_size2), where (im_size1, im_size2) is an image
    size.
    Performs FFT of every input image and puts it in an array bigx of size
    (n_im, im_size1, im_size2, 2), where "2" represents real and imaginary
    dimensions
    :param folder: path to the folder, which contains images
    :param n_im: number of images to load from the folder
    :param normalize: if True - the xbig data will be normalized
    :param imrotate: if True - the each input image will be rotated by 90, 180,
    and 270 degrees
    :return:
    bigx: 4D array of frequency data of size (n_im, im_size1, im_size2, 2)
    bigy: 3D array of images of size (n_im, im_size1, im_size2)
    
    
    Modified by Chong Duan, 10/17/2018
    """

#    # Initialize the arrays:
#    if imrotate:  # number of images is 4 * n_im
#        bigy = np.empty((n_im * 4, 64, 64))
#        bigx = np.empty((n_im * 4, 64, 64, 2))
#    else:
#        bigy = np.empty((n_im, 64, 64))
#        bigx = np.empty((n_im, 64, 64, 2))

#    im = 0  # image counter
    bigy = []
    filenames = os.listdir(folder)
    for filename in filenames[n_cases[0]:n_cases[1]]:
        if not filename.startswith('.'):
            temp = loadmat(os.path.join(folder, filename))['res']
            
            # Clean the STONE sense recon data
            row, col, t1w, sli = temp.shape
            temp = np.reshape(temp, (row, col, -1))
            valid_mask = (np.abs(np.squeeze(temp[int(row/2), int(col/2), :])) != 0)
            final_images = temp[:,:,valid_mask]
            
#            # Resize images
            final_images = np.abs(final_images)
            final_images_resized = np.zeros((64,64,final_images.shape[2]))
            for i in range(final_images.shape[2]):
                final_images_resized[:,:,i] = cv2.resize(final_images[:,:,i], (64,64))
            
#            # Only take a small part of the data
#            final_images = final_images[140:180,140:180,:]
            
#            # Convert to abs values
#            final_images = np.abs(final_images)
#            
#            # Normalize based on single patient case
#            final_images = (final_images - np.mean(final_images)) / np.std(final_images)
            
#            bigy_temp = cv2.imread(os.path.join(folder, filename),
#                                   cv2.IMREAD_GRAYSCALE)
            
            
            bigy.append(final_images_resized)
    
    bigy = np.asarray(bigy)
    cases, row, col, imgs = bigy.shape
    bigy = np.transpose(np.reshape(np.transpose(bigy, (1,2,3,0)), (row, col, -1)), (2,0,1))
    
    # convert to k-space
    imgs, row, col = bigy.shape
    bigx = np.empty((imgs, row, col, 2))
    for i in range(imgs):
        bigx[i, :, :, :] = create_x(np.squeeze(bigy[i,:,:]), normalize=False)
    
    # convert bigx from complex to abs values
    bigy = np.abs(bigy)
    
#            im += 1
#            if imrotate:
#                for angle in [90, 180, 270]:
#                    bigy_rot = im_rotate(bigy_temp, angle)
#                    bigx_rot = create_x(bigy_rot, normalize)
#                    bigy[im, :, :] = bigy_rot
#                    bigx[im, :, :, :] = bigx_rot
#                    im += 1

#        if imrotate:
#            if im > (n_im * 4 - 1):  # how many images to load
#                break
#        else:
#            if im > (n_im - 1):  # how many images to load
#                break

#    if normalize:
#        bigx = (bigx - np.amin(bigx)) / (np.amax(bigx) - np.amin(bigx))

    return bigx, bigy


def create_x_motion(y, normalize=False):
    """
    Prepares frequency data from image data: first image y is padded by 8
    pixels of value zero from each side (y_pad_loc1), then second image is
    created by moving the input image (64x64) 8 pixels down -> two same images
    at different locations are created; then both images are transformed to
    frequency space and their frequency space is combined as if the image
    moved half-way through the acquisition (upper part of freq space from one
    image and lower part of freq space from another image)
    expands the dimensions from 3D to 4D, and normalizes if normalize=True
    :param y: input image
    :param normalize: if True - the frequency data will be normalized
    :return: "Motion corrupted" frequency-space data of the input image,
    4D array of size (1, im_size1, im_size2, 2), third dimension (size: 2)
    contains real and imaginary part
    """

    # Pad y and move 8 pixels
    y_pad_loc1 = np.zeros((80, 80))
    y_pad_loc2 = np.zeros((80, 80))
    y_pad_loc1[8:72, 8:72] = y
    y_pad_loc2[0:64, 8:72] = y

    # FFT of both images
    img_f1 = np.fft.fft2(y_pad_loc1)  # FFT
    img_fshift1 = np.fft.fftshift(img_f1)  # FFT shift
    img_f2 = np.fft.fft2(y_pad_loc2)  # FFT
    img_fshift2 = np.fft.fftshift(img_f2)  # FFT shift

    # Combine halfs of both k-space - as if subject moved 8 pixels in the
    # middle of acquisition
    x_compl = np.zeros((80, 80), dtype=np.complex_)
    x_compl[0:41, :] = img_fshift1[0:41, :]
    x_compl[41:81, :] = img_fshift2[41:81, :]

    # Finally, separate into real and imaginary channels
    x_real = x_compl.real
    x_imag = x_compl.imag
    x = np.dstack((x_real, x_imag))

    x = np.expand_dims(x, axis=0)

    if normalize:
        x = x - np.mean(x)

    return x

def create_x(y, motion=False):
    """
    Prepares frequency data from image data: applies to_freq_space,
    expands the dimensions from 3D to 4D, and normalizes if normalize=True
    :param y: input image
    :param normalize: if True - the frequency data will be normalized
    :return: frequency data 4D array of size (1, im_size1, im_size2, 2)
    """
    
    if motion:
        # Pad y and move 8 pixels
        y_pad_loc1 = np.zeros((80, 80))
        y_pad_loc2 = np.zeros((80, 80))
        y_pad_loc1[8:72, 8:72] = y
        y_pad_loc2[0:64, 8:72] = y
        
        # FFT of both images
        img_f1 = np.fft.fft2(y_pad_loc1)  # FFT
        img_fshift1 = np.fft.fftshift(img_f1)  # FFT shift
        img_f2 = np.fft.fft2(y_pad_loc2)  # FFT
        img_fshift2 = np.fft.fftshift(img_f2)  # FFT shift
        
        # Combine halfs of both k-space - as if subject moved 8 pixels in the
        # middle of acquisition
        x_compl = np.zeros((80, 80), dtype=np.complex_)
        x_compl[0:41, :] = img_fshift1[0:41, :]
        x_compl[41:81, :] = img_fshift2[41:81, :]
        
        # Finally, separate into real and imaginary channels
        x_real = x_compl.real
        x_imag = x_compl.imag
        x = np.dstack((x_real, x_imag))
        
        x = np.expand_dims(x, axis=0)
    else: 
        x = to_freq_space(y)
        x = np.expand_dims(x, axis=0)

    return x


def to_freq_space(img):
    """ Performs FFT of an image
    :param img: input 2D image
    :return: Frequency-space data of the input image, third dimension (size: 2)
    contains real ans imaginary part
    """

    img_f = np.fft.fft2(img)  # FFT
    img_fshift = np.fft.fftshift(img_f)  # FFT shift
    img_real = img_fshift.real  # Real part: (im_size1, im_size2)
    img_imag = img_fshift.imag  # Imaginary part: (im_size1, im_size2)
    img_real_imag = np.dstack((img_real, img_imag))  # (im_size1, im_size2, 2)

    return img_real_imag


def im_rotate(img, angle):
    """ Rotates an image by angle degrees
    :param img: input image
    :param angle: angle by which the image is rotated, in degrees
    :return: rotated image
    """
    rows, cols = img.shape
    rotM = cv2.getRotationMatrix2D((cols/2-0.5, rows/2-0.5), angle, 1)
    imrotated = cv2.warpAffine(img, rotM, (cols, rows))

    return imrotated


'''
# For debugging: show the images and their frequency space

dir_temp = 'path to folder with images'
X, Y = load_images_from_folder(dir_temp, 5, normalize=False, imrotate=True)

print(Y.shape)
print(X.shape)


plt.subplot(221), plt.imshow(Y[12, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(Y[13, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(Y[14, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(Y[15, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()

X_m = 20*np.log(np.sqrt(np.power(X[:, :, :, 0], 2) +
                        np.power(X[:, :, :, 1], 2)))  # Magnitude
plt.subplot(221), plt.imshow(X_m[12, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(X_m[13, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(X_m[14, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(X_m[15, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
'''
