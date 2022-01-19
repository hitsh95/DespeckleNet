import torch
from .base_model import BaseModel
from . import complex_networks
from . import Loss
from torch.nn import functional as F

''' based on GAN '''

class DespeckleModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):        
        parser.set_defaults(netG='complex_unet', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        """Initialize the DespeckleModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1',  'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_BI', 'real_BI', 'real_BP', 'fake_BP'] 
        
        if self.isTrain:
            self.model_names = ['G',  'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        
        self.netG = complex_networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = complex_networks.define_D(opt.input_nc + 2, opt.ndf, opt.netD,             #######################################
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.alpha = 0.6
        self.test_loss_log = 0
        self.test_loss = torch.nn.L1Loss()
        if self.isTrain:
            # define loss functions
            self.criterionGAN = Loss.GANLoss(opt.gan_mode).to(self.device)
            if not opt.mixloss:
                self.criterionL1 = torch.nn.L1Loss()
            else:
                self.criterionL1 = Loss.MixedPix2PixLoss().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        x = torch.split(self.real_B, 1, dim=1)
        self.real_BI= x[0]
        self.real_BP= x[1]
        
    def forward(self):#############################################################################
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""        
        self.fake_BI, self.fake_BP, self.feat_out = self.netG(self.real_A)  # G(A)
        self.test_loss_log = self.test_loss(self.fake_BI, self.real_BI) + self.test_loss(self.fake_BP, self.real_BP)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_BI, self.fake_BP), 1)  ############ 
        pred_fake = self.netD(fake_AB.detach()) 
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_BI, self.real_BP), 1)    ########### 
        pred_real = self.netD(real_AB)                
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_BI, self.fake_BP), 1) ###########
        pred_fake = self.netD(fake_AB)   
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = self.criterionL1(self.fake_BI, self.real_BI) + self.criterionL1(self.fake_BP, self.real_BP) 
   
        self.loss_G_L1 = loss_G_L1  * 100
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L1 + self.loss_G_GAN
        self.loss_G.backward()
 
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

