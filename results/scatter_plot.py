from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat
import PIL.Image as Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import csv
def mae(img1, img2):
    mae = np.mean( abs(img1 - img2)  )
    return mae 

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.weight'] = 'bold'

color = ['Greens','Reds','Blues','gray']

root = '/home/lius/sh/pix2pix_ls/results/'
path = '/dif2IP/test_latest/images/'
algs = ['complex', 'optica1', 'optica2', 'real']
show_names = ['_fake_BI', '_real_BI','_real_A','_fake_BP', '_real_BP']
ext = ['.mat', '.png']



file_list = [item.split('/')[-1].split('_real')[0] for item in glob("/home/lius/sh/pix2pix_ls/results/ckp_simu3_real/dif2IP/test_latest/images/*_real_A.png")]
ssim_I, ssim_P = {},{}
mse_I, mse_P  = {},{}
for alg in algs:
    ssim_I[alg], mse_I[alg] = [],[]
    ssim_P[alg], mse_P[alg] = [],[]
    for scatter_size in range(5, 15, 2):
        tmp_ssim_i, tmp_ssim_p,tmp_mse_i,tmp_mse_p = 0,0,0,0
        i = 0
        for file_name in file_list:
            i += 1
            print(file_name)
            gt_P = loadmat(root+'ckp_simu3_real/dif2IP/test_latest/images/'+ file_name + show_names[-1] + ext[0])['phase']
            gt_P = gt_P[:,:,0]
            gt_I = cv2.imread(root+ 'ckp_simu3_real/dif2IP/test_latest/images/'+ file_name + show_names[1] + ext[1])
            gt_I = gt_I[:,:,0] /255.

            P = loadmat(root+ 'ckp_simu'+str(scatter_size)+'_'+alg+'/dif2IP/test_latest/images/'+ file_name + show_names[-2] + ext[0])['phase']
            P = P[:,:,0]
       
            I = cv2.imread(root+'ckp_simu'+str(scatter_size)+'_'+alg+'/dif2IP/test_latest/images/'+ file_name + show_names[0] + ext[1])
            I = I[:,:,0] /255.

            tmp_ssim_i += ssim(gt_I, I)
            tmp_ssim_p += ssim(gt_P, P)
            tmp_mse_i += mae(gt_I, I)
            tmp_mse_p += mae(gt_P, P)  # MAE
        
        ssim_I[alg].append(tmp_ssim_i / i) 
        ssim_P[alg].append(tmp_ssim_p / i)
        mse_I[alg].append(tmp_mse_i / i)
        mse_P[alg].append(tmp_mse_p / i)
        
with open('simu_ssim_mae_'+alg+'.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow('ssim_I')
    for alg in algs:
        writer.writerow(ssim_I[alg])
    
    writer.writerow('ssim_P')
    for alg in algs:
        writer.writerow(ssim_P[alg])
    
    writer.writerow('mse_I')
    for alg in algs:
        writer.writerow(mse_I[alg])

    writer.writerow('mse_p')
    for alg in algs:
        writer.writerow(ssim_P[alg])
        











