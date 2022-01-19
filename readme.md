# DeepdespeckeNet

## Workflow overview
![Workflow overview](/assets/overview.jpg)


## Network structure
![Network structure](/assets/network.jpg)

## Prerequisties
Before training or testing, you need to prepare the deep-learning environment:
```
Python 3.9
CPU or NVIDIA GPU + CUDA CuDNN
```

## Environment Preparing
You should prepare at least 2 2080ti gpus or change the batch size in training process. 
```
pip install -r requirement.txt
```

## Dataset preparing
We provide representitive testing dataset due to the limits of the upload file size, including the Hela cells with Petri dish and the nasopharyngeal carcinoma cell with thin tape. Please put these datasets on ../datasets/test/xxx
```
For training dataset, Please put the dataset on ../datasets/train/xxx
```
## Training process

Run 
```
python3 train.py --dataroot hela_cell --gpu_ids 0,1  --checkpoints_dir ./checkpoints/ckp_hela_complex --batch_size 16  --ngf 64  --no_flip
```
## Testing process

For Hela cell despeckle results, run:
```
python3 test.py --dataroot hela_cell  --checkpoints_dir ./checkpoints/ckp_hela_complex  --ngf 64 --results_dir ./results/ckp_hela_complex
```

For nasopharyngeal_carcinoma_cell despeckle results, run:
```
python3 test.py --dataroot nasopharyngeal_carcinoma_cell  --checkpoints_dir ./checkpoints/ckp_C666_complex  --ngf 64 --results_dir ./results/ckp_C666_complex
```

The default results are saved in ./results/xxx/dif2IP/test_latest/images, where xxx_fake_BI.bmp and xxx_fake_BP.mat are the outputs of our network, xxx_real_BI.bmp and xxx_real_BP.mat are the ground truths, and the xxx_real_A.bmp is the speckle image. You can also open ./results/xxx/dif2IP/test_latest/images/index.html to check the entire results.


Here we present the despeckle results of two representative datasets c666-1 and Hela. For more results please see the manuscript.