import numpy as np
import torch
import PIL.Image as Image
import time
import cv2
from scipy.io import loadmat
import math
import os
import shutil
from tensorboardX import SummaryWriter


def generate_random_phase():
    p = np.random.rand(512, 512)
    p = np.where(p<0.7, 0., p)
    gauss_kernel = gauss(5, 1)
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

def main(k1, mode):
    
    path_old = '/data/lipeng/simu/' + mode 
    path_I = '/data/lipeng/simu/I/'
    path_P = '/data/lipeng/simu/P/'
    # os.makedirs(path_I)
    # os.makedirs(path_P)
    root = '/data/lipeng/simu_scatter'+str(round(512/k1))+'/'   #########################################################################################################
    
    path_new = root + mode +'/'
    if not os.path.exists(path_new):
        os.makedirs(path_new)
        
    start = time.time()

    plan = "diff_prop/"
    project = "project1/"
    mode = "train"  # "train", 'valid", or "test"

    # path_1 = "/data/caijh/diffuser/origin/" + mode + "_amp/"
    # # path_1 = "/data/caijh/diffuser/origin/train_amp/train_amp1_00000.bmp"
    # path_2 = "/data/caijh/diffuser/origin/" + mode + "_phase/"

    n = 0

    for each in os.listdir(path_old):#########################################################################################################
        n += 1
        print(n, each)
        file_name_1 = path_I + each
        
        w = 512
        dot = np.random.rand(round(w/k1), round(w/k1)) * 2 * np.pi
        dot = np.cos(dot) + 1j * np.sin(dot)
        h = dot.shape[0]
        scatter = np.zeros((512, 512)) + 1j * np.zeros((512, 512))
        scatter[:h, :h] = dot

        fstart = abs(np.fft.fft2(scatter))**2 
        ma = fstart.max()
        mi = fstart.min()
        Istart = (fstart-mi)/(ma-mi)+ 0.01

        temp_1 = Image.open(file_name_1)
        temp_1 = temp_1.convert('L')  # gray
        org_amp = np.array(temp_1).astype(np.float32)
        amp = np.zeros((512, 512), dtype=np.float32)
        amp[110:-110, 110:-110] = org_amp[110:-110, 110:-110]
        
        dif = Istart * amp
        dif = np.clip(dif, 0, 255)

        cv2.imwrite(path_new + each, dif)#########################################################################################################

if __name__ == '__main__':
    # k = [40, 46, 60, 73, 100, 180]
    k = [46, 60, 73, 100]
    for i, ki in enumerate(k):
       main(ki, 'dif')
       main(ki, 'test')
       print('_____________________________scater size: '+str(round(512/ki))+' finished__________________________________')