# DeepdespeckeNet

### Workflow overview
![Workflow overview](/assets/overview.jpg)


### Network structure
![Network structure](/assets/network.jpg)

## Environment Preparing
```
python 3.9
```
You should prepare at least 2 2080ti gpus or change the batch size. 


### Training process

Run 

python3 train.py --dataroot 0513 --gpu_ids 0,1,2,3  --checkpoints_dir ./checkpoints/ckp_0513_complex --batch_size 16  --ngf 64  --no_flip


### Testing process

Run

python3 test.py --dataroot 0513  --checkpoints_dir ./checkpoints/ckp_0513_complex  --ngf 64 --results_dir ./results/ckp_0513_complex


