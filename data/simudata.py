import numpy as np
from numpy import random
import torch
import PIL.Image as Image
import time
import cv2
from scipy.io import loadmat
import math
import os
import shutil
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

def generate_random_phase():
    p = np.random.rand(512, 512)
    p = np.where(p<0.1, 0., p)
    gauss_kernel = gauss(7, 1)
    fl_cv = cv2.filter2D(p, -1, gauss_kernel)
    return fl_cv


def gauss(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size//2
    if sigma<=0:
        sigma = ((kernel_size-1)*0.5-1)*0.3+0.8
    
    s = sigma**2
    sum_val =  0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i-center, j-center
            
            kernel[i, j] = np.exp(-(x**2+y**2)/2*s)
            sum_val += kernel[i, j]
    
    kernel = kernel/sum_val
    
    return kernel


def BandLimitTransferFunction(pixelsize, z, lamda, Fvv, Fhh):

    hSize, vSize = Fvv.shape
    dU = (np.float64(hSize) * pixelsize) ** -1.0
    dV = (np.float64(vSize) * pixelsize) ** -1.0
    Ulimit = ((2.0 * dU * z) ** 2.0 + 1.0) ** -0.5 / lamda
    Vlimit = ((2.0 * dV * z) ** 2.0 + 1.0) ** -0.5 / lamda
    freqmask = ((Fvv ** 2.0 / (Ulimit ** 2.0) + Fhh ** 2.0 * (lamda ** 2.0)) <= 1.0) & ((Fvv ** 2.0 * (lamda ** 2.0) + Fhh ** 2.0 / (Vlimit ** 2.0)) <= 1.0)

    return freqmask

def PropGeneral(Fhh, Fvv, lamda, refidx, z):

    lamdaeff = lamda / refidx
    DiffLimMat = np.where(1.0/(lamdaeff**2.0) <= Fhh ** 2.0 + Fvv ** 2.0, 0., 1.0)

    temp1 = 2.0 * np.pi * z / lamdaeff
    temp1 = np.complex(temp1)
    temp2 = (1.0 - (lamdaeff * Fvv) ** 2.0 - (lamdaeff * Fhh) ** 2.0) ** 0.5
    temp2 = temp2 + 1j *0.
    H_t = np.exp(1j * temp1 * temp2)

    H = np.where(DiffLimMat == 0, np.zeros(H_t.shape, dtype=np.complex), H_t)

    return H

def propagate(img, pixelsize, refidx, lamda, z, convunits=True, zeropad=True, freemask=True):
    if convunits:
        lamda = lamda * 1e-9
        pixelsize = pixelsize * 1e-6
        z = z * 1e-6

    nv, nh = img.shape
    spectrum = np.fft.fft2(img)
    spectrum = np.fft.fftshift(spectrum)

    nfv, nfh = spectrum.shape

    fs = 1/pixelsize
    Fh = fs / nfh * np.arange((-np.ceil((nfh - 1) / 2)), np.floor((nfh - 1) / 2) + 0.5,
                                    dtype=np.float64)
    Fv = fs / nfv * np.arange((-np.ceil((nfv - 1) / 2)), np.floor((nfv- 1) / 2) + 0.5,
                                    dtype=np.float64)
    [Fhh, Fvv] = np.meshgrid(Fh, Fv)
    H = PropGeneral(Fhh, Fvv, lamda, refidx, z)
    freqmask = BandLimitTransferFunction(pixelsize, z, lamda, Fvv, Fhh)
    
    spectrum_z = np.multiply(spectrum, H)
    if freemask:
        spectrum_z = np.multiply(spectrum_z, freqmask + 1j * 0.)

    spectrum_z = np.fft.ifftshift(spectrum_z)
    img_z = np.fft.ifft2(spectrum_z)
    img_z = img_z[:nv, :nh]
    return img_z, H

def main():
    path_dif = '/data/lipeng/simu/dif'
    path_test = '/data/lipeng/simu/test'
    path_I = '/data/lipeng/simu/I/'
    path_P = '/data/lipeng/simu/P/'
    # os.makedirs(path_I)
    # os.makedirs(path_P)
    root = '/data/lipeng/simu_70/'   #########################################################################################################
    
    path_dif_new = root + "dif/"
    path_test_new = root + "test/"
    if not os.path.exists(path_dif_new):
        os.makedirs(path_dif_new)
        os.makedirs(path_test_new)
    start = time.time()
    
    # random_phase = loadmat('./data/random_phase')['phase']
    random_phase = generate_random_phase()

    plan = "diff_prop/"
    project = "project1/"
    mode = "train"  # "train", 'valid", or "test"

    # path_1 = "/data/caijh/diffuser/origin/" + mode + "_amp/"
    # # path_1 = "/data/caijh/diffuser/origin/train_amp/train_amp1_00000.bmp"
    # path_2 = "/data/caijh/diffuser/origin/" + mode + "_phase/"

    n = 0

    for each in os.listdir(path_dif):#########################################################################################################
        n += 1
        print(n, each)
        file_name_1 = path_I + each
        file_name_2 = path_P + each.replace('amp', 'angle')
     
    # file_name_1 = '/data/caijh/diffuser/origin/train_amp/train_amp1_00000.bmp'
    # file_name_2 = '/data/caijh/diffuser/origin/train_phase/train_angle1_00000.bmp'
        temp_1 = Image.open(file_name_1)
        temp_1 = temp_1.convert('L')  # gray
        org_amp = np.array(temp_1).astype(np.float32)
        amp = np.zeros((512, 512), dtype=np.float32)
        amp[110:-110, 110:-110] = org_amp[110:-110, 110:-110]
        
        temp_2 = Image.open(file_name_2)
        temp_2 = temp_2.convert('L')
        org_phase = np.array(temp_1).astype(np.float32)
        phase = np.zeros((512, 512), dtype=np.float32)
        phase[110:-110, 110:-110] = org_phase[110:-110, 110:-110]
        phase = phase/255.0 * 2 * np.pi - np.pi

        # 把array转换成tf.float32
        
        input_image = amp * np.cos(phase) + 1j * amp * np.sin(phase)
        pixel_size_const = 0.3733
        refidx_const = 1.0
        lamda_const = 405.0
        distance_const = 50. 
        distance_out = 70.0  #########################################################################################################20

        output, H = propagate(input_image, pixel_size_const, refidx_const,
                            lamda_const, distance_const, True, False, True)

        output_abs = np.absolute(output)
        output_phase = np.zeros((output_abs.shape[0], output_abs.shape[1]))
        # r_phase = np.load(random_phase_path)# must "" not ''
        output_angle = np.angle(output)
        # output_phase = output_angle
        output_phase = random_phase/2 + output_angle
        output = output_abs * np.cos(output_phase) + 1j * output_abs * np.sin(output_phase)

        output_backprop, H = propagate(output, pixel_size_const, refidx_const,
                                    lamda_const, distance_out, True, False, True)

        # cv2.imwrite('./data/I.bmp', amp)
        input_amp = np.absolute(output_backprop)
        # cv2.imwrite('./data/I.bmp', input_amp)
        cv2.imwrite(path_dif_new + each, input_amp)#########################################################################################################

if __name__ == '__main__':
    # 
    main()
