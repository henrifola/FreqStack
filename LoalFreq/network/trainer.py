import torch
import torch.nn as nn
from .LLnet import LLnet
import os

class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')

        self.model = LLnet(wave=opt.wavelet_type, window_size=opt.window_size).to(self.device)
        self.total_steps = 0
        self.lr = opt.lr

        if opt.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()

            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=opt.lr,
                    betas=(opt.beta1 if hasattr(opt, 'beta1') else 0.9, 0.999),
                    weight_decay=opt.weight_decay if hasattr(opt, 'weight_decay') else 0
                )
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=opt.lr,
                    momentum=opt.momentum if hasattr(opt, 'momentum') else 0.0,
                    weight_decay=opt.weight_decay if hasattr(opt, 'weight_decay') else 0
                )
            else:
                raise ValueError("optim should be [adam, sgd]")

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float().view(-1)

    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.view(-1), self.label)

    def optimize_parameters(self):
        self.forward()
        loss = self.get_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss.item()

    def adjust_learning_rate(self, factor=0.8, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group['lr'] * factor, min_lr)
            param_group['lr'] = new_lr
        self.lr = self.optimizer.param_groups[0]['lr']

    def save_networks(self, epoch_label):
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, f'{epoch_label}_model.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved model to: {save_path}")
