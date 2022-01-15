import importlib


from pathlib import Path
import logging
import time
import numpy as np
import torch

def creat_logger(opt, mode='train'):
    root_output_dir = Path(opt.output_dir)  # output
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    
    model = opt.model_name  
    final_output_dir = root_output_dir / model
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True) # output/model
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(mode, time_str)
    final_log_file = final_output_dir / log_file  # output/model/train_time.log
    

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tblog_dir = Path('tb_log') / model / time_str
    print('=> creating {}'.format(tblog_dir))
    tblog_dir.mkdir(parents=True, exist_ok=True)   # tb_log/model/time

    return logger,  str(final_output_dir), str(final_log_file), str(tblog_dir), time_str


def tensor2im(input_image, mode):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        # if is_I:
        #     image_numpy = np.transpose(image_numpy, (1, 2, 0))  * 255.0
        # else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if mode == 'no':
            ma, mi = image_numpy.max(), image_numpy.min()
            image_numpy = (image_numpy-mi)/(ma-mi)
        image_numpy = image_numpy * 255.0  # post-processing: tranpose and scaling
        # elif label == 'real_BI'  or label == 'fake_BI':
        #     image_numpy = image_numpy * 255.0
        # elif label == 'real_BP' or label == 'fake_BP':
        #     image_numpy = image_numpy * np.pi
        #     ma, mi = image_numpy.max(), image_numpy.min()
        #     image_numpy = (image_numpy - mi)/(ma-mi) * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy

def save_current_visual(visu, epoch, iter, writer, phase):
    '''save debug results to tensorboard'''
    # modes = ['-1,1'] * 5
    # i = 0
    for name, tensor in visu.items():
        # img = tensor2im(name, tensor)
        save_name = phase + '_epoch_' + str(epoch) + '_' + str(iter) +'_'+name

        if len(tensor.shape) == 3:
            dataformats = 'HW'
        else:
            dataformats = 'CHW'

        if name in ['real_BP','fake_BP']:
            img_np = tensor[0].detach().cpu().float().numpy()
            ma, mi = img_np.max(), img_np.min()
            img_np = (img_np-mi)/(ma-mi)
            writer.add_image(save_name, img_np, epoch, dataformats=dataformats)
        else:
            img_np = tensor[0].detach().cpu().float().numpy()
            img_np = (img_np+1)/2
            writer.add_image(save_name, img_np, epoch, dataformats=dataformats)
        # if modes[i] == 'no': # convert to 0-1
        #     img_np = tensor[0].detach().cpu().float().numpy()
        #     ma, mi = img_np.max(), img_np.min()
        #     img_np = (img_np-mi)/(ma-mi)
        #     writer.add_image(save_name, img_np, epoch, dataformats=dataformats)
        # elif modes[i] == '-1,1':
        #     img_np = tensor[0].detach().cpu().float().numpy()
        #     img_np = (img_np+1)/2
        #     writer.add_image(save_name, img_np, epoch, dataformats=dataformats)
        # else:
        #     writer.add_image(save_name, tensor[0], epoch)  # c*h*w [0,1]   dataformats : CHW, HWC, HW.
        '''https://pytorch.org/docs/1.7.1/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_image'''
        # i += 1 
        
        #         writer.add_image(save_name, tensor[0,j].unsqueeze(0), epoch)  # c*h*w [0,1]
            
