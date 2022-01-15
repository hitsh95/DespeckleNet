# from util.visualizer import cal_ssim
import glob
import os
from PIL import Image
from numpy.core import multiarray
from scipy.io import loadmat
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error, normalized_root_mse
import numpy as np
import csv

# results = ['ckp_simu5_complex','ckp_simu7_complex','ckp_simu9_complex']

# gts = ['simu/simu_scatter5/test', 'simu/simu_scatter7/test', 'simu/simu_scatter9/test']
results = [
    'ckp_simu5_complex', 
    'ckp_simu7_complex', 
    'ckp_simu9_complex', 
    'ckp_simu11_complex', 
    'ckp_simu13_complex']

gts = [
    'simu/simu_scatter5/test', 
    'simu/simu_scatter7/test', 
    'simu/simu_scatter9/test', 
    'simu/simu_scatter11/test', 
    'simu/simu_scatter13/test', 
]

# results = ['ckp_0330_complex', 'ckp_0330_real', \
#     'ckp_0513_complex', 'ckp_0513_real', 'ckp_0513_optica1', 'ckp_0513_optica2', \
#     'ckp_1018_complex', 'ckp_1018_real', 'ckp_her2_complex',  \
#     'ckp_simu5_complex', 'ckp_simu5_real', 'ckp_simu5_optica1', 'ckp_simu5_optica2', \
#     'ckp_simu7_complex', 'ckp_simu7_real', 'ckp_simu7_optica1', 'ckp_simu7_optica2', \
#     'ckp_simu9_complex', 'ckp_simu9_real', 'ckp_simu9_optica1', 'ckp_simu9_optica2',
#     'ckp_simu11_complex', 'ckp_simu11_real', 'ckp_simu11_optica1', 'ckp_simu11_optica2',
#     'ckp_simu13_complex', 'ckp_simu13_real', 'ckp_simu13_optica1', 'ckp_simu13_optica2'
# ]

# gts = ['0330/test', '0330/test','0513_sh/test', '0513_sh/test', \
#     '0513_sh/test', '0513_sh/test', '1018/test', '1018/test', 'her2/test', \
#     'simu/simu_scatter5/test', 'simu/simu_scatter5/test', 'simu/simu_scatter5/test', 'simu/simu_scatter5/test', \
#     'simu/simu_scatter7/test', 'simu/simu_scatter7/test', 'simu/simu_scatter7/test', 'simu/simu_scatter7/test', \
#     'simu/simu_scatter9/test', 'simu/simu_scatter9/test', 'simu/simu_scatter9/test', 'simu/simu_scatter9/test', \
#     'simu/simu_scatter11/test', 'simu/simu_scatter11/test', 'simu/simu_scatter11/test', 'simu/simu_scatter11/test', \
#     'simu/simu_scatter13/test', 'simu/simu_scatter13/test', 'simu/simu_scatter13/test', 'simu/simu_scatter13/test'
# ]

def cal_ssim(real_I, fake_I, real_P, fake_P):
    I_mse = mean_squared_error(real_I, fake_I)
    P_mse = mean_squared_error(real_P, fake_P)
    I_nrmse = normalized_root_mse(real_I, fake_I, normalization='min-max')
    P_nrmse = normalized_root_mse(real_P, fake_P, normalization='min-max')
    I_ssim = structural_similarity(real_I, fake_I,multichannel=True) 
    P_ssim = structural_similarity(real_P, fake_P,multichannel=True)
    I_psnr = peak_signal_noise_ratio(real_I, fake_I)
    P_psnr = peak_signal_noise_ratio(real_P, fake_P)
    return I_ssim, P_ssim, I_psnr, P_psnr, I_mse, P_mse, I_nrmse, P_nrmse

if __name__ == '__main__':
    I_ssim_list, P_ssim_list = [], []
    I_psnr_list, P_psnr_list = [], []
    I_mse_list, P_mse_list = [], []
    I_nrmse_list, P_nrmse_list = [], []
    for result, gt in zip(results, gts):
        dataroot = '/data/shahao/' + gt
        resultroot = './results/' + result +'/dif2IP/test_latest/images/'
        
        test_path = sorted(glob.glob(os.path.join(dataroot, '*' + '.bmp')))
        indexs = [img.split('/')[-1].split('.')[0] for img in test_path]

        if len(indexs) == 0:
            print("there is no data in testdata! please check ", result)
            break

        I_real_path = [resultroot + id + '_real_BI.png' for id in indexs]
        P_real_path = [resultroot + id + '_real_BP.png' for id in indexs]
        I_fake_path = [resultroot + id + '_fake_BI.png' for id in indexs]
        P_fake_path = [resultroot + id + '_fake_BP.png' for id in indexs]

        I_ssim, P_ssim = 0, 0
        I_psnr, P_psnr = 0, 0
        I_mse, P_mse = 0, 0
        I_nrmse, P_nrmse = 0,0

        for i in range(len(indexs)):
            real_I = np.array(Image.open(I_real_path[i]))
            # real_P = loadmat(P_real_path[i])['phase_save']
            real_P = np.array(Image.open(P_real_path[i]))

            fake_I = np.array(Image.open(I_fake_path[i]))
            # fake_P = loadmat(P_fake_path[i])['phase']
            fake_P = np.array(Image.open(P_fake_path[i]))
            I_s, P_s, I_p, P_p, I_m, P_m, I_r, P_r = \
                 cal_ssim(real_I, fake_I, real_P[:,:,0], fake_P[:,:,0])
            I_ssim += I_s
            P_ssim += P_s
            I_psnr += I_p
            P_psnr += P_p
            I_mse += I_m
            P_mse += P_m
            I_nrmse += I_r
            P_nrmse += P_r

        I_ssim /= len(indexs)
        P_ssim /= len(indexs)
        I_psnr /= len(indexs)
        P_psnr /= len(indexs)
        I_mse /= len(indexs)
        P_mse /= len(indexs)
        I_nrmse /= len(indexs)
        P_nrmse /= len(indexs)

        I_ssim_list.append(I_ssim)
        P_ssim_list.append(P_ssim)
        I_psnr_list.append(I_psnr)
        P_psnr_list.append(P_psnr)
        I_mse_list.append(I_mse)
        P_mse_list.append(P_mse)
        I_nrmse_list.append(I_nrmse)
        P_nrmse_list.append(P_nrmse)

        print(result)
        print(I_ssim, I_psnr, I_mse, I_nrmse)
        print(P_ssim, P_psnr, P_mse, P_nrmse)

    with open("./results/ssim_results.csv","w") as csvfile: 
        writer = csv.writer(csvfile)        
        for i in range(len(I_ssim_list)):
            writer.writerow([results[i]])
            writer.writerow(["I_ssim","I_psnr","I_mse", "P_ssim", "P_psnr", "P_mse", "I_nrmse", "P_nrmse", "I_cor", "P_cor"])
            writer.writerow([I_ssim_list[i], P_ssim_list[i], I_psnr_list[i], P_psnr_list[i], I_mse_list[i], 
              P_mse_list[i], I_nrmse_list[i], P_nrmse_list[i]])

    

