from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """
    

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        # parser.add_argument('--results_dir', type=str, default='./results/0113_complexv1_doublechannel_block9_linear_base0330/', help='saves results here.')
        parser.add_argument('--results_dir', type=str, default='./results/ckp_tissue_laser_complexdouble/', help='saves results here.')


        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=100, help='how many test images to run')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.isTrain = False
        return parser
