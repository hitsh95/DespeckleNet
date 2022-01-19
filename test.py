"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models):
    Test a DespeckleNet model (both sides):
        python test.py --dataroot 0513  --checkpoints_dir ./checkpoints/ckp_0513_complex  --ngf 64

    The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

See options/base_options.py and options/test_options.py for more test options.
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, save_feats
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    issim, imse, pssim, pmse =0, 0, 0, 0
    with open(f'{opt.results_dir}/test_log.txt', 'a') as f:
        f.write('------------------%s\n' % opt.results_dir)
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
           print('processing (%04d)-th image... %s' % (i, img_path))
        iss, im, ps, pm = save_images(webpage, visuals, img_path, opt.dataroot, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        issim += iss
        imse += im
        pssim += ps
        pmse += pm
    issim /= len(dataset)
    imse /= len(dataset)
    pssim /= len(dataset)
    pmse /= len(dataset)
    with open(f'{opt.results_dir}/test_log.txt','a') as f:
        f.write("avg_I_ssim: %f \tavg_I_mse: %f \tavg_P_ssim: %f \t avg_P_mse: %f \n" % (issim, imse, pssim, pmse))

    webpage.save()  # save the HTML
