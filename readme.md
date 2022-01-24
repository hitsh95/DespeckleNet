# DeepdespeckeNet

## Workflow overview
![Workflow overview](/assets/overview.jpg)


## Network structure
![Network structure](/assets/network.jpg)

## System requirements
Before training or testing, you need to prepare the deep-learning environment:
```
Python 3.9
CPU or NVIDIA GPU + CUDA CuDNN
Linux OS
```

## Installation guide
You should prepare at least 2 2080ti gpus or change the batch size in training process. 
```
pip install -r requirement.txt
```
Installation will be completed quickly

## Demo
You can download part of the datasets on [Google Drive](https://drive.google.com/drive/folders/1jF5DLO8Ug0hElA0rOrNcWN_RJUkaqjCp?usp=sharing), and we will open souce the entire data soon.

**For testing dataset, please put it on "../datasets/test/xxx"**

**For training dataset, please put it on "../datasets/train/xxx"**

Due to the upload file size limitation, we provide representative despeckle image pairs and weight files over the c666-1 dataset in the zip file. Unzip it and run: 

```                          
python3 test.py --dataroot nasopharyngeal_carcinoma_cell  --checkpoints_dir ./checkpoints/ckp_C666_complex  --ngf 64 --results_dir ./results/ckp_C666_complex
```

Running time is about 20s when using a single 2080Ti graphics card. The default results are saved in "./results/xxx/dif2IP/test_latest/images", where xxx_fake_xxx are the outputs of our network, xxx_real_xxx are the ground truths, and the xxx_real_A.bmp is the speckle image. You can also open "./results/xxx/dif2IP/test_latest/images/index.html" to check the results.


## Instruction for use
### Training process

Run 
```
python3 train.py --dataroot xxx --gpu_ids 0,1  --checkpoints_dir ./checkpoints/xxx --batch_size 16  --ngf 64  --no_flip
```
the traing dataroot should be one of the [breast_cancer_cell, hela_cell, breast_cancer_tissue, nasopharyngeal_carcinoma_cell, simu_scatter4,simu_scatter6,simu_scatter8, simu_scatter10, simu_scatter12]

You can download the pre-trained weights files on [Google Drive](https://drive.google.com/drive/folders/1-KcDxA5QWE8G-_YlJphAYG-20pO4x5rx?usp=sharing). Put the weight file on "./checkpoints/".
### Testing process

Run:
```
python3 test.py --dataroot xxx  --checkpoints_dir ./checkpoints/xxx  --ngf 64 --results_dir ./results/xxx
```

The testing dataroot should be one of the [breast_cancer_cell, hela_cell, breast_cancer_tissue, breast_cancer_tissue_2048, nasopharyngeal_carcinoma_cell, simu_scatter4,simu_scatter6,simu_scatter8, simu_scatter10, simu_scatter12]



## Common problem
When you encounter any problems please contact us by email <ybzhang08@hit.edu.cn> or <shahao@stu.hit.edu.cn> 
