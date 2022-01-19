import os
from scipy.io.matlab.mio import savemat
from data.base_dataset import BaseDataset, get_params, get_transform, transform_phase

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import glob
import random

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.dataroot = opt.dataroot

        if self.opt.phase == 'train':
            if self.dataroot not in ['simu_scatter5', 'simu_scatter7', 'simu_scatter9', 'simu_scatter11', 'simu_scatter13']:
                self.dif = sorted(glob.glob(os.path.join('../datasets/train/'+self.dataroot+'/dif', '*' + '.bmp')))
                print('../datasets/train/'+self.dataroot+'/dif', '*' + '.bmp')
                self.I = [img.replace('dif', 'I') for img in self.dif]
                self.P = [img.replace('dif', 'P').replace('bmp', 'mat') for img in self.dif]

            else:
                self.dif = sorted(glob.glob(os.path.join('../datasets/train/'+self.dataroot+'/dif', '*' + '.bmp')))
                self.I = [img.replace('dif', 'I') for img in self.dif]
                self.P = [img.replace('dif', 'P').replace('amp','angle') for img in self.dif]
         
        else:
            # all_data
            if self.dataroot not in ['simu_scatter5', 'simu_scatter7', 'simu_scatter9', 'simu_scatter11', 'simu_scatter13', \
                'living_hela_video', 'breast_cancer_tissue_2048']:
                self.dif = sorted(glob.glob(os.path.join('../datasets/test/'+self.dataroot+'/dif', '*' + '.bmp')))
                self.I = [img.replace('dif', 'I') for img in self.dif]
                self.P = [img.replace('dif', 'P').replace('bmp', 'mat') for img in self.dif]
            elif self.dataroot in ['simu_scatter5', 'simu_scatter7', 'simu_scatter9', 'simu_scatter11', 'simu_scatter13']:
                self.dif = sorted(glob.glob(os.path.join('../datasets/test/'+self.dataroot+'/dif', '*' + '.bmp')))
                self.I = [img.replace('dif', 'I') for img in self.dif]
                self.P = [img.replace('dif', 'P').replace('amp','angle') for img in self.dif]
            elif self.dataroot in ['living_hela_video', 'breast_cancer_tissue_2048']:
                self.dif = sorted(glob.glob(os.path.join('../datasets/test/'+self.dataroot+'/dif', '*' + '.bmp')))
                self.I = self.dif
                self.P = self.dif # no ground truth in living_hela_video and breast_cancer_tissue_2048 datasets
            else:
                print(self.dataroot, 'not in any test datasets!')

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = 1
        self.output_nc = 1

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index

        if index >= len(self.dif):
            index = index - len(self.dif)
     
        A_path = self.dif[index]
        B1_path = self.I[index]
        B2_path = self.P[index]
        # contrast enhancing
        img_dif = cv2.imread(A_path, cv2.IMREAD_GRAYSCALE) / 255.
        img_dif = np.power(img_dif, 0.3) * 255.
        img_dif = img_dif.astype(np.uint8)
        A = Image.fromarray(img_dif)
        B1 = Image.open(B1_path)

        if self.dataroot not in ['simu_scatter5', 'simu_scatter7', 'simu_scatter9', 'simu_scatter11', 'simu_scatter13', \
                'living_hela_video', 'breast_cancer_tissue_2048']:
            B2 = loadmat(B2_path)['phase_save']
        else:
            B2 = Image.open(B2_path)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, (512,512))
        transform = get_transform(self.opt, transform_params, grayscale=True)    # with normalize
        
        A = transform(A)
        B1 = transform(B1)
        
        if self.dataroot not in ['simu_scatter5', 'simu_scatter7', 'simu_scatter9', 'simu_scatter11', 'simu_scatter13', \
                'living_hela_video', 'breast_cancer_tissue_2048']:
            B2 = torch.from_numpy(B2).type(torch.FloatTensor).unsqueeze(dim=0) 
        else:
            B2 = transform(B2)

        B = torch.cat((B1,B2), 0)
        if self.opt.led_input:
            return {'A': B1, 'B': B2, 'A_paths': A_path, 'B_paths': A_path}
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': A_path}

        
    def __len__(self):
        """Return the total number of images in the dataset."""
        return min(len(self.dif), 4000)

