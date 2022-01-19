"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., DespeckeNet, UNet, IDiffNet) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    python train.py --dataroot 0513 --gpu_ids 0,1,2,3  --checkpoints_dir ./checkpoints/ckp_0513_complex --batch_size 16  --ngf 64  --no_flip

"""
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
import os
from util.visualizer import save_images, cal_ssim
from util.myutil import save_current_visual

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    opt_val = TestOptions().parse()

    opt_val.batch_size = 1
    opt_val.phase = 'val'
    dataset_val = create_dataset(opt_val)
    print('The number of validation = %d' % len(dataset_val))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    tb_log_dir = os.path.join('tb_log',opt.checkpoints_dir.split('/')[-1])
    if not os.path.join(tb_log_dir):
        os.makedirs(tb_log_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    writer = writer_dict['writer']
    best_effi = 1e6
    best_Issim, best_Pssim =  0, 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    # for epoch in range(opt.epoch_count, opt.n_epochs + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        log = epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0  :   # display images on visdom and save images to a HTML file
               save_result = total_iters % opt.update_html_freq == 0
               model.compute_visuals()
               visuals = model.get_current_visuals()
               save_current_visual(visuals, epoch, epoch_iter, writer, phase='train')

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                
                global_steps = writer_dict['train_global_steps']
                # writer.add_scalar('P_loss', losses['GP_L1'], global_steps)
                writer.add_scalar('L1_loss', losses['G_L1'], global_steps)
                # writer.add_scalar('I_defocal_loss', losses['GI_defocal_L1'], global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            iter_data_time = time.time()
        
        l1loss_log = 0
        I_ssim, P_ssim = 0, 0
        for i,data in enumerate(dataset_val):
            if i >= opt_val.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            l1loss_log += model.test_loss_log.item()
            I_s, P_s = cal_ssim(visuals)
            I_ssim += I_s
            P_ssim += P_s
        if epoch>100 and epoch % 10 == 0:
            save_current_visual(visuals, epoch, 0, writer,phase='valid')
        l1loss_log /= len(dataset_val)
        I_ssim /= len(dataset_val)
        P_ssim /= len(dataset_val)

        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_L1loss', l1loss_log, global_steps)
        writer.add_scalar('I_ssim', I_ssim, global_steps)
        writer.add_scalar('P_ssim', P_ssim, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
        if l1loss_log < best_effi:
            best_effi = l1loss_log
            best_Issim = I_ssim
            best_Pssim = P_ssim
            print('saving the best model epoch %d' % epoch)
            save_suffix = 'best'
            model.save_networks(save_suffix)
        print('--------------------------epoch {:d}\t I_ssim: {:.4f}\t P_ssim: {:.4f}'.format(log, I_ssim, P_ssim))
        print('--------------------------current best I_ssim: {:.4f}\t best P_ssim: {:.4f}'.format(best_Issim, best_Pssim))
        # if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        # if epoch % 50 == 0:
        #     model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
