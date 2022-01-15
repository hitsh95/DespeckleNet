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
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths

        self.opt = opt
        self.dataroot = opt.dataroot

        if self.opt.phase == 'train':
            # TWO kinds data
            if self.dataroot not in ['0513', 'simu']:# 0411-org 0420-hela 0513-hela-live   
                self.dif = sorted(glob.glob(os.path.join('/data/shahao/'+self.dataroot+'/dif', '*' + '.bmp')))
                self.I = [img.replace('dif', 'I') for img in self.dif]
                P = [img.replace('dif', 'P') for img in self.dif]
                self.P = [img.replace('bmp', 'mat') for img in P]
            elif self.dataroot in ['0513']:                
                #### special process for living hela dataset
                dir = '/data/shahao/0513/dif'
                list_file = os.listdir(dir)
                self.dif = []
                for ai in list_file:
                    sp = ai.split('_')
                    group = sp[0]
                    if group in ['5', '6', '11', '12', '14', '17','1', '2', '3', '4', '7']:
                        self.dif.append(os.path.join(dir, ai)) 
                self.I = [img.replace('dif', 'I') for img in self.dif]
                P = [img.replace('dif', 'P') for img in self.dif]
                self.P = [img.replace('bmp', 'mat') for img in P]
              
            elif self.dataroot in ['simu']: # simulate data
                self.dif = sorted(glob.glob("/data/shahao/simu/simu_scatter5"+"/dif/*.bmp")) #train_input_amp1_00000.npy
                self.I = [img.replace('simu_scatter5/dif', 'simu/I') for img in self.dif]  #train_amp/train_amp1_00000.bmp
                self.P = [img.replace('I','P').replace('amp', 'angle') for img in self.I]   #train_phase/train_angle1_00000.bmp   

        else:
            # all_data
            if self.dataroot not in ['0513' , 'simu', '0513_video', '0330_crop']:# 0411-org 0420-hela 0513-hela-live   0516-bad
                self.dif = sorted(glob.glob(os.path.join('/data/shahao/'+self.dataroot+'/test', '*' + '.bmp')))
                self.I = [img.replace('test', 'I') for img in self.dif]
                P = [img.replace('test', 'P') for img in self.dif]
                self.P = [img.replace('bmp', 'mat') for img in P]
            elif self.dataroot in ['0330_crop']:
                self.dif = sorted(glob.glob(os.path.join('/data/shahao/'+self.dataroot, '*' + '_dif.png')))
                self.I = [img.replace('dif', 'I') for img in self.dif]
                self.P = [img.replace('dif', 'P') for img in self.dif]
            elif self.dataroot in ['0513']: # hela
                self.dif = []             

                dir = '/data/shahao/0513_new/test'
                list_file = os.listdir(dir)
                for ai in list_file:
                    sp = ai.split('_')
                    group = sp[0]
                    if group in ['5', '6', '11', '12', '14', '17']:
                        self.dif.append(os.path.join(dir, ai))
                self.I = [img.replace('test', 'I') for img in self.dif]
                P = [img.replace('test', 'P') for img in self.dif]
                self.P = [img.replace('bmp', 'mat') for img in P]
            elif self.dataroot == 'simu':
                self.dif = sorted(glob.glob("/data/shahao/simu/simu_scatter11"+"/test/*.bmp")) #train_input_amp1_00000.npy
                self.I = [img.replace('simu_scatter11/test', 'simu/I') for img in self.dif]   #train_amp/train_amp1_00000.bmp
                self.P = [img.replace('I','P').replace('amp', 'angle') for img in self.I]   #train_phase/train_angle1_00000.bmp

            elif self.dataroot in ['0513_video']:                
                self.dif = sorted(glob.glob(os.path.join('/data/shahao/'+self.dataroot+'/test_5', '*' + '.bmp')))
                self.I = self.dif
                self.P = self.dif
            else:
                print(self.dataroot, 'not in any test dataset!')

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

        if self.dataroot not in ['simu', '0513_video','0330_crop']:
            B2 = loadmat(B2_path)['phase_save']
        else:
            B2 = Image.open(B2_path)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, (512,512))
        transform = get_transform(self.opt, transform_params, grayscale=True)    # with normalize
        
        A = transform(A)
        B1 = transform(B1)
        
        if self.dataroot not in ['0513_video', 'simu', '0330_crop']:
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

