
import torch
import torch.nn as nn
import math
from torchvision import models
from torchvision.models.vgg import vgg16
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(1,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

###############################################################################
# Helper Functions
###############################################################################
class Gradient_Net(nn.Module):
    def __init__(self):
        super(Gradient_Net, self).__init__()
        self.p1d = [1, 1, 1, 1]
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        x = F.pad(x, self.p1d, mode='reflect')
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        ext = torch.zeros_like(x)
        x = torch.cat((ext[:,0,:,:].unsqueeze_(1), x), 1)
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out

class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.register_buffer("kernel", self._cal_gaussian_kernel(11, 1.5))
        self.L = 2.0
        self.k1 = 0.01
        self.k2 = 0.03

    @staticmethod
    def _cal_gaussian_kernel(size, sigma):
        g = torch.Tensor([math.exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)])
        g = g / g.sum()
        window = g.reshape([-1, 1]).matmul(g.reshape([1, -1]))
        kernel = torch.reshape(window, [1, 1, size, size]).repeat(2, 1, 1, 1)
        return kernel

    def forward(self, img0, img1):
        """
        :param img0: range in (-1, 1)
        :param img1: range in (-1, 1)
        :return: SSIM loss i.e. 1 - ssim
        """
        mu0 = torch.nn.functional.conv2d(img0, self.kernel, padding=0, groups=1)
        mu1 = torch.nn.functional.conv2d(img1, self.kernel, padding=0, groups=1)
        mu0_sq = torch.pow(mu0, 2)
        mu1_sq = torch.pow(mu1, 2)
        var0 = torch.nn.functional.conv2d(img0 * img0, self.kernel, padding=0, groups=1) - mu0_sq
        var1 = torch.nn.functional.conv2d(img1 * img1, self.kernel, padding=0, groups=1) - mu1_sq
        covar = torch.nn.functional.conv2d(img0 * img1, self.kernel, padding=0, groups=1) - mu0 * mu1
        c1 = (self.k1 * self.L) ** 2
        c2 = (self.k2 * self.L) ** 2
        ssim_numerator = (2 * mu0 * mu1 + c1) * (2 * covar + c2)
        ssim_denominator = (mu0_sq + mu1_sq + c1) * (var0 + var1 + c2)
        ssim = ssim_numerator / ssim_denominator
        ssim_loss = 1.0 - ssim
        return ssim_loss

class olLoss(torch.nn.Module):
    def __init__(self):
        super(olLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, img0, img1):
        """
        
        """
        img0_IP = torch.split(img0, 1, 1)
        img0_OL1 = img0_IP[0] * torch.cos(img0_IP[1])
        # img0_OL2 = img0_IP[0] * torch.sin(img0_IP[1])

        img1_IP = torch.split(img1, 1, 1)
        img1_OL1 = img1_IP[0] * torch.cos(img1_IP[1])
        # img1_OL2 = img1_IP[0] * torch.sin(img1_IP[1])

        ol_loss = self.loss(img0_OL1, img1_OL1) 
        return ol_loss

class MixedPix2PixLoss(torch.nn.Module):
    def __init__(self):
        super(MixedPix2PixLoss, self).__init__()
        self.cha_loss = CharbonnierLoss()
        # self.edge_loss = EdgeLoss()
        self.alpha = 0.05
        self.ssim_loss = SSIMLoss()
        # self.l1_loss = torch.nn.L1Loss()
        # self.ol_loss = olLoss()
        # # self.vgg = Vgg16()
        # self.grad = Gradient_Net()  

    def forward(self, pred, target):
        """
        :param pred: (bs, c, h, w) image ranging in (-1, 1)
        :param target: (bs, c, h, w) image ranging in (-1, 1)
        :param reduce: (str) reduction method, "mean" or "none" or "sum"
        :return:
        """
        # preds = pred.split(1, 1)
        # predI, predP = preds[0], preds[1]
        # targets = target.split(1, 1)
        # targetI, targetP = targets[0], targets[1]
        
        ssim_loss = torch.mean(self.ssim_loss(pred, target))
        # l1_loss = self.l1_loss(predI, targetI) + 3 * self.l1_loss(predP, targetP)
        l1_loss = self.cha_loss(pred, target) 
        # edg_loss = self.edge_loss(pred, target)

        # # loss_grad
        # mask0 = torch.zeros_like(target)
        # mask1 = torch.ones_like(target)
        # grad = self.grad(target)
        # mask = torch.where(grad > 0.12, mask1, mask0)  # 0.03
        # loss_grad = self.l1_loss(mask*pred, mask*target)

        # loss_percep
        # y_feats = self.vgg(target)
        # y_hat_feats = self.vgg(pred)
        # loss_percep = self.l1_loss(y_feats[1], y_hat_feats[1])
        
        weighted_mixed_loss = self.alpha * ssim_loss + (1.0 - self.alpha) * l1_loss 
        # weighted_mixed_loss =  l1_loss 
        return weighted_mixed_loss

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
