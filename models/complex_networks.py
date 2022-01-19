import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.functional import interpolate

###############################################################################
# Helper Functions
###############################################################################
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=1e-6)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], led=False):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: complex_unet | Unet | IDiffNet
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        ComplexUNet: our modified U-Net(with 5 real-valued Residual blocks and 4 complex-valued Residual blocks). This model is mainly based on
        the pix2pix model used in image translation task.
         The original pix2pix paper: https://arxiv.org/abs/1611.07004

        UNet generator: Li's despeckle network (optica1).
         The original paper: https://www.osapublishing.org/optica/fulltext.cfm?uri=optica-5-10-1181&id=398582
        IDiffNet generator: Barbastathis's despeckle network (optica2).
         The original paper: https://www.osapublishing.org/optica/fulltext.cfm?uri=optica-5-7-803&id=395106
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'UNet':
        net = Deep_speckle(input_nc, output_nc * 2)
    elif netG == 'IDiffNet':
        net = IDiffNet(input_nc, output_nc * 2)
    elif netG == 'complex_unet':
        net  = ComplexUnet(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):

    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70x70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class conv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(conv_2d, self).__init__()
        pad = kernel_size // 2 
        # self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad), nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.conv(x) 
class basic(nn.Module):
    def __init__(self, in_channels=128, feats=32):
        super(basic, self).__init__()
        self.res0 = nn.Sequential(conv_2d(in_channels, feats, 3, 1),nn.LeakyReLU(0.2, True), 
                                    conv_2d(feats, feats, 3, 1),nn.LeakyReLU(0.2, True))
        self.res1 = nn.Sequential(conv_2d(feats, feats, 3, 1),nn.LeakyReLU(0.2, True), 
                                  conv_2d(feats, feats, 3, 1))
        self.res2 = nn.Sequential(conv_2d(feats, feats, 3, 1),nn.LeakyReLU(0.2, True), 
                                  conv_2d(feats, feats, 3, 1))
        self.last = nn.Sequential(conv_2d(feats, feats, 3, 1),nn.LeakyReLU(0.2, True), 
                                  nn.Conv2d(feats, 11, 3, 1, 1), nn.ReLU())

    def forward(self, x): 
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.last(x)
        return x

## final version
class ComplexUnet(nn.Module):
    def __init__(self,input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(ComplexUnet, self).__init__()
        self.down1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.down_relu1 = nn.ReLU(True)
        self.down2 = nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1)
        self.norm2 = norm_layer(ngf*2)
        self.down_relu2 = nn.ReLU(True)
        self.down3 = nn.Conv2d(ngf*2, ngf * 4, kernel_size=4, stride=2, padding=1)
        self.norm3 = norm_layer(ngf*4)
        self.down_relu3 = nn.ReLU(True)
        self.down4 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.norm4 = norm_layer(ngf*8)
        self.down_relu4 = nn.ReLU(True)
        self.down5_r = nn.Conv2d(ngf * 8, ngf*8, kernel_size=4, stride=2, padding=1)
        # self.down5_i = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)

        self.submodelI = nn.ModuleList()
        for _ in range(5):
            self.submodelI.append(realResnetBlock(ngf*8, 'reflect', norm_layer=norm_layer, use_dropout=False, use_bias=True))
        for _ in range(4):
            self.submodelI.append(complexResnetBlock(ngf*8, 'reflect', norm_layer=norm_layer, use_dropout=False, use_bias=True))

        # intensity
        self.up1_conv = nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4, stride=2,padding=1, bias=True)
        self.up1_norm = norm_layer(ngf * 8)
        self.up_relu2 = nn.LeakyReLU(0.2, True)
        self.up2_conv = nn.ConvTranspose2d(ngf * 8*2 , ngf * 4, kernel_size=4, stride=2, padding=1, bias=True)
        self.up2_norm = norm_layer(ngf * 4)
        self.up_relu3 = nn.LeakyReLU(0.2, True)
        self.up3_conv = nn.ConvTranspose2d(ngf * 4*2, ngf * 2, kernel_size=4, stride=2, padding=1, bias=True)
        self.up3_norm = norm_layer(ngf * 2)
        self.up_relu4 = nn.LeakyReLU(0.2, True)
        self.up4_conv = nn.ConvTranspose2d(ngf * 2*2 , ngf , kernel_size=4, stride=2, padding=1, bias=True)
        self.up4_norm = norm_layer(ngf)
        self.up_relu5 = nn.LeakyReLU(0.2, True)
        self.up5_conv = nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=4, stride=2, padding=1, bias=True)
        self.Iout = nn.Tanh()
                                  
        # phase
        self.up1_convp = nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=True)
        self.up1_normp = norm_layer(ngf * 8)
        self.up_relu2p = nn.LeakyReLU(0.2, True)
        self.up2_convp = nn.ConvTranspose2d(ngf * 8*2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=True)
        self.up2_normp = norm_layer(ngf * 4)
        self.up_relu3p = nn.LeakyReLU(0.2, True)
        self.up3_convp = nn.ConvTranspose2d(ngf * 4*2, ngf * 2, kernel_size=4, stride=2, padding=1, bias=True)
        self.up3_normp = norm_layer(ngf * 2)
        self.up_relu4p = nn.LeakyReLU(0.2, True)
        self.up4_convp = nn.ConvTranspose2d(ngf * 2*2, ngf, kernel_size=4, stride=2, padding=1, bias=True)
        self.up4_normp = norm_layer(ngf)
        self.up_relu5p = nn.LeakyReLU(0.2, True)
        self.up5_convp = nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1, bias=True)
        self.Pout = nn.Tanh()

    def forward(self, input_r):                                    # 1
        inte = 4
        feats = []
        feats_i = []
        feats_p = []
        feats.append(input_r[0,0,:,:])
        d1 = self.down_relu1(self.down1(input_r))          # n
        feats.append(d1[0,::inte,:,:])
        d2 = self.down_relu2(self.norm2(self.down2(d1)))    # 2n
        feats.append(d2[0,::inte,:,:])
        d3 = self.down_relu3(self.norm3(self.down3(d2)))   # 4n
        feats.append(d3[0,::inte,:,:])
        d4 = self.down_relu4(self.norm4(self.down4(d3)))   # 8n
        feats.append(d4[0,::inte,:,:])
        res = self.down5_r(d4)
        feats.append(res[0,::inte,:,:])
        # res_i = self.down5_i(d4)
  
        for block in self.submodelI:
            res = block(res)
        feats.append(res[0,::inte,:,:])
        
        u1_I = self.up1_norm(self.up1_conv(res))     # 8n
        u1_I= self.up_relu2(torch.cat([u1_I, d4],1))
        feats_i.append(u1_I[0,::inte,:,:])
        u2_I = self.up2_norm(self.up2_conv(u1_I))     # 4n
        u2_I = self.up_relu3(torch.cat([u2_I, d3], 1))
        feats_i.append(u2_I[0,::inte,:,:])
        u3_I = self.up3_norm(self.up3_conv(u2_I))     # 2n
        u3_I = self.up_relu4(torch.cat([u3_I, d2], 1))
        feats_i.append(u3_I[0,::inte,:,:])
        u4_I = self.up4_norm(self.up4_conv(u3_I))     # n
        u4_I = self.up_relu5(torch.cat([u4_I, d1], 1))
        feats_i.append(u4_I[0,::inte,:,:])
        I = self.Iout(self.up5_conv(u4_I))           # 1
        feats_i.append(I[0,0,:,:])

        u1_P = self.up1_normp(self.up1_convp(res))     # 8n
        u1_P= self.up_relu2p(torch.cat([u1_P, d4],1))
        feats_p.append(u1_P[0,::inte,:,:])
        u2_P = self.up2_normp(self.up2_convp(u1_P))     # 4n
        u2_P = self.up_relu3p(torch.cat([u2_P, d3], 1))
        feats_p.append(u2_P[0,::inte,:,:])
        u3_P = self.up3_normp(self.up3_convp(u2_P))     # 2n
        u3_P = self.up_relu4p(torch.cat([u3_P, d2], 1))
        feats_p.append(u3_P[0,::inte,:,:])
        u4_P = self.up4_normp(self.up4_convp(u3_P))     # n
        u4_P = self.up_relu5p(torch.cat([u4_P, d1], 1))
        feats_p.append(u4_P[0,::inte,:,:])
        # P = self.Pout(self.up5_convp(u4_P))           # 1    
        P = self.up5_convp(u4_P)          # 1
        feats_p.append(P[0,0,:,:])

        feat_all = [feats, feats_i, feats_p]
        return I,P,feat_all
     

class Conv_factory(nn.Module):
    def __init__(self,input_nc, output_nc, dropout_rate=0.5):
        super(Conv_factory, self).__init__()
        block = [nn.BatchNorm2d(input_nc), nn.ReLU(True), nn.Conv2d(input_nc, output_nc, 5, 1, 4, 2)]
        if dropout_rate:
            block.append(nn.Dropout2d(0.5))
        self.block = nn.Sequential(*block)
    
    def forward(self, x):
        return self.block(x)


class Dense(nn.Module):
    def __init__(self, input_nc, n_layer, growth_rate):
        super(Dense,self).__init__()
        self.n_layer = n_layer
        self.block = nn.ModuleList()
        cur_nc = input_nc
        for j in range(n_layer):
            self.block.append(Conv_factory(cur_nc, growth_rate))
            cur_nc = input_nc + (j+1)*growth_rate
    def forward(self, x):
        for j in range(self.n_layer):
            x_dense = self.block[j](x)
            x = torch.cat([x, x_dense], 1)
        return x

# optical article1
class Deep_speckle(nn.Module):
    def __init__(self, input_nc, output_nc, growth_rate=16):
        super(Deep_speckle,self).__init__()
        self.output_nc = output_nc
        self.conv1 = nn.Conv2d(input_nc, 64, 3, padding=1)
        self.relu1 = nn.ReLU(True)
        self.db1 = Dense(64, n_layer=4, growth_rate=16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(128,128,3,padding=1)
        self.relu2 = nn.ReLU(True)
        self.db2 = Dense(128, 4, 16)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128+64, 256,3,padding=1)
        self.relu3 = nn.ReLU(True)
        self.db3 = Dense(256, 4, 16)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256+64, 512,3,padding=1)
        self.relu4 = nn.ReLU(True)
        self.db4 = Dense(512, 4, 16)  # db4
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(512+64, 1024,3,padding=1)
        self.relu5 = nn.ReLU(True)
        self.db5 = Dense(1024, 4, 16)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor= 2)

        self.conv6 = nn.Conv2d(1024+64, 512,3,padding=1)  
        self.relu6 = nn.ReLU(True)  # --------concat db4
        self.conv7 = nn.Conv2d(1024+64, 512,3,padding=1)  
        self.relu7 = nn.ReLU(True)  # up5
        self.db7 = Dense(512, 3,16)
        self.up7 = nn.UpsamplingBilinear2d(scale_factor= 2)

        self.conv8 = nn.Conv2d(512+48, 256,3,padding=1)  
        self.relu8 = nn.ReLU(True)  # --------concat db3
        self.conv9 = nn.Conv2d(512+64, 256,3,padding=1)  
        self.relu9 = nn.ReLU(True)  # 
        self.db9 = Dense(256, 3,16)
        self.up9 = nn.UpsamplingBilinear2d(scale_factor= 2)

        self.conv10 = nn.Conv2d(256+48, 128,3,padding=1)  
        self.relu10 = nn.ReLU(True)  # --------concat db2
        self.conv11 = nn.Conv2d(256+64, 128,3,padding=1)  
        self.relu11 = nn.ReLU(True)  # 
        self.db11 = Dense(128, 3,16)
        self.up11 = nn.UpsamplingBilinear2d(scale_factor= 2)

        self.conv12 = nn.Conv2d(128+48, 64,3,padding=1)  
        self.relu12 = nn.ReLU(True)  # --------concat db1
        self.conv13 = nn.Conv2d(128+64, 64,3,padding=1)  
        self.relu13 = nn.ReLU(True)  # 
        self.db13 = Dense(64, 3,16)
        
        self.conv14 = nn.Conv2d(64+48, 32,3,padding=1)  
        self.relu14 = nn.ReLU(True) 
        self.conv15 = nn.Conv2d(32, output_nc,3,padding=1)  
        self.relu15 = nn.Tanh()  # 

    def forward(self, x):
        db1 = self.db1(self.relu1(self.conv1(x)))
        pool1 = self.pool1(db1)  # 128
        
        db2 = self.db2(self.relu2(self.conv2(pool1)))
        pool2 = self.pool2(db2)  # 192
        
        db3 = self.db3(self.relu3(self.conv3(pool2)))
        pool3 = self.pool3(db3)  # 320

        db4 = self.db4(self.relu4(self.conv4(pool3)))
        pool4 = self.pool4(db4)  # 576

        db5 = self.db5(self.relu5(self.conv5(pool4)))

        db6 = self.relu6(self.conv6(self.up5(db5)))
        db7 = self.db7(self.relu7(self.conv7(torch.cat([db6, db4],1))))

        db8 = self.relu8(self.conv8(self.up7(db7)))
        db9 = self.db9(self.relu9(self.conv9(torch.cat([db8, db3],1))))

        db10 = self.relu10(self.conv10(self.up9(db9)))
        db11 = self.db11(self.relu11(self.conv11(torch.cat([db10, db2],1))))

        db12 = self.relu12(self.conv12(self.up11(db11)))
        db13 = self.db13(self.relu13(self.conv13(torch.cat([db12, db1],1))))

        db14 = self.relu14(self.conv14(db13))
        out = self.conv15(db14)

        nc = self.output_nc // 2
        I = out[:,:nc,:,:]
        P = out[:,nc:,:,:]

        feat = []

        return I,P,feat

class DenseDownBlock(nn.Module):
    def __init__(self, inc, ouc, dropout = 0.5):
        super(DenseDownBlock,self).__init__()
        block = [Dense(inc, 3, 12), nn.BatchNorm2d(inc+36), nn.ReLU(True), nn.Conv2d(inc+36, ouc, 1,1), nn.Dropout(dropout), nn.AvgPool2d(2,2)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class DenseUpBlock(nn.Module):
    def __init__(self, inc, ouc, dropout = 0.5):
        super(DenseUpBlock,self).__init__()
        block = [nn.UpsamplingBilinear2d(scale_factor=2),nn.Conv2d(inc, inc, 3,1,1),nn.ReLU(True),  Dense(inc,3,12), nn.Conv2d(inc+36, ouc, 3,1,1), nn.ReLU(True)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

# optical article2
class IDiffNet(nn.Module):
    def __init__(self, input_nc, output_nc, growth_rate=16):
        super(IDiffNet,self).__init__()
        self.output_nc = output_nc
        self.conv1 = nn.Conv2d(input_nc, 16, 3, padding=1)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.AvgPool2d(2, 2)    #64
        
        self.ddb1 = DenseDownBlock(16, 26)  # 32
        self.ddb2 = DenseDownBlock(26, 31)  # 16
        self.ddb3 = DenseDownBlock(31, 33)  # 8
        self.ddb4 = DenseDownBlock(33, 34)  # 4
        self.ddb5 = DenseDownBlock(34, 35)  # 2
        self.ddb6 = DenseDownBlock(35, 35)  # 1
        
        self.dm = nn.Conv2d(35,36, 1)

        self.ud1 = DenseUpBlock(36, 36)   # concat ddb5
        self.ud2 = DenseUpBlock(35+36, 36) # concat ddb4
        self.ud3 = DenseUpBlock(34+36, 36) # concat ddb3
        self.ud4 = DenseUpBlock(33+36, 36) # concat ddb2
        self.ud5 = DenseUpBlock(31+36, 36) # concat ddb1
        self.ud6 = DenseUpBlock(26+36, 36) # concat pool1
        
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv14 = nn.Conv2d(16+36, 36,3,padding=1)  
        self.relu14 = nn.ReLU(True) 
        self.conv15 = nn.Conv2d(36, output_nc,3,padding=1)  
        self.relu15 = nn.Tanh()  # 

    def forward(self, x):
        pool1 = self.pool1(self.relu1(self.conv1(x)))
        ddb1 = self.ddb1(pool1)
        ddb2 = self.ddb2(ddb1)
        ddb3 = self.ddb3(ddb2)
        ddb4 = self.ddb4(ddb3)
        ddb5 = self.ddb5(ddb4)
        ddb6 = self.ddb6(ddb5)

        up1=self.ud1(self.dm(ddb6))
        up2 = self.ud2(torch.cat([up1, ddb5], 1))
        up3 = self.ud3(torch.cat([up2, ddb4], 1))
        up4 = self.ud4(torch.cat([up3, ddb3], 1))
        up5 = self.ud5(torch.cat([up4, ddb2], 1))
        up6 = self.ud6(torch.cat([up5, ddb1], 1))
        up7 = self.relu14(self.conv14(self.upsamp(torch.cat([up6, pool1],1))))

        out = self.conv15(up7)

        nc = self.output_nc // 2
        I = out[:,:nc,:,:]
        P = out[:,nc:,:,:]
        feat = []
        # out = self.relu15(self.conv15(up7))

        return I, P, feat

class realResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(realResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),  nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class ComplexConv2d(nn.Module):
    def __init__(self, ic, nc, k_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_G = nn.Conv2d(ic, nc, k_size, stride, padding, dilation, 2, bias)
        self.conv_math= nn.Conv2d(ic, nc, 1, stride, 0, dilation, 1, bias)
        self.GN = nn.GroupNorm(2, nc)
        self.IN = nn.InstanceNorm2d(nc)

    def forward(self, input):
        return self.IN(self.conv_math(self.GN(self.conv_G(input))))
class complexResnetBlock(nn.Module):
    """Define a complexResnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, is_last=False):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(complexResnetBlock, self).__init__()
        self.is_last = is_last
        
        conv_block = [nn.ReflectionPad2d(1), ComplexConv2d(dim, dim, k_size=3, bias=True),nn.LeakyReLU(0.2, True),
                      nn.ReflectionPad2d(1),ComplexConv2d(dim, dim, k_size=3, bias=True)
                            ]
        self.convs = nn.Sequential(*conv_block)
        

    def forward(self, x):
        """Forward function (with skip connections)"""
        return self.convs(x) + x

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult  # 2
            nf_mult = min(2 ** n, 8)  # 4
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult # 4
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)




