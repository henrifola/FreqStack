# base_options.py
import argparse
import os
import time
import torch

class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Train LLnet (minimal options)")
        self.initialized = False

    def initialize(self):
        p = self.parser
        p.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
        p.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
        p.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizers')

        
        p.add_argument('--wavelet_type', type=str, default='haar', help='Type of wavelet used in LLnet (e.g., haar, db1, etc.)')
        p.add_argument('--window_size', type=int, default=32, help='Sliding window size for local patch processing')


        p.add_argument('--classes', nargs='*', default=[], help='Image classes to train on (default: all)')

        # Model and run name
        p.add_argument('--name', type=str, default='llnet', help='Experiment name')
        p.add_argument('--model_name', type=str, default='llnet', help='Model filename used by dynamic loader')
        p.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Directory to save logs and checkpoints')

        # Dataset
        p.add_argument('--dataroot', type=str, default='./data', help='Root path to dataset folder')
        p.add_argument('--train_split', type=str, default='train', help='Training subfolder name')
        p.add_argument('--val_split', type=str, default='val', help='Validation subfolder name')

        # Image preprocessing
        p.add_argument('--loadSize', type=int, default=256, help='Resize images to this size before cropping')
        p.add_argument('--cropSize', type=int, default=224, help='Crop size for images')
        p.add_argument('--no_crop', action='store_true', help='Disable cropping')
        p.add_argument('--no_resize', action='store_true', help='Disable resizing')
        p.add_argument('--no_flip', action='store_true', help='Disable random horizontal flipping')
        p.add_argument('--rz_interp', default='bilinear', help='Resize interpolation method')

        # Training
        p.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
        p.add_argument('--niter', type=int, default=20, help='Number of training epochs')
        p.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
        p.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer type')
        
        p.add_argument('--loss_freq', type=int, default=25, help='Log training loss every X steps')
        p.add_argument('--delr_freq', type=int, default=10, help='Reduce LR every X epochs')

        # Data loading
        p.add_argument('--num_threads', type=int, default=4, help='Number of threads for data loading')
        p.add_argument('--serial_batches', action='store_true', help='Disable data shuffling')

        # Class balancing and mode
        p.add_argument('--mode', type=str, default='binary', help='Dataset mode: binary or filename')
        # p.add_argument('--classes', nargs='*', default=[], help='Image classes to train on (default: all)')
        p.add_argument('--class_bal', action='store_true', help='Use class-balanced sampling')

        # GPU and resume
        p.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='List of GPU IDs')
        p.add_argument('--isTrain', action='store_true', help='Use training mode')
        p.add_argument('--continue_train', action='store_true', help='Resume from latest checkpoint')
        p.add_argument('--epoch', type=str, default='latest', help='Which checkpoint to load [latest or int]')
        
        
        
        p.add_argument('--model_path', type=str, default=None, help='Path to trained model checkpoint (.pth)')

        

        self.initialized = True

    def parse(self, print_options=True):
        if not self.initialized:
            self.initialize()

        opt = self.parser.parse_args()
        opt.name = opt.name + "_" + time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # GPU setup
        if torch.cuda.is_available() and opt.gpu_ids:
            torch.cuda.set_device(opt.gpu_ids[0])

        # Logging
        if print_options:
            print('--- Training Options ---')
            for k, v in sorted(vars(opt).items()):
                print(f'{k}: {v}')
            print('------------------------')

        return opt
