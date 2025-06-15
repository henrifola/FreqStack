import os
import time
import torch
from data import create_dataloader
from validate import validate
from options.base_options import BaseOptions
from network.trainer import Trainer
from tensorboardX import SummaryWriter
import numpy as np

# Test domains (used for final evaluation)
vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, target_ap=0.99, target_acc=0.99):
        self.patience = patience
        self.min_delta = min_delta
        self.target_ap = target_ap
        self.target_acc = target_acc
        self.best_loss = None
        self.wait = 0

    def check(self, val_loss, val_ap, val_acc):
        if val_ap >= self.target_ap and val_acc >= self.target_acc:
            if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.wait = 0
            else:
                self.wait += 1
                print(f" No improvement in val loss ({val_loss:.4f}) vs best ({self.best_loss:.4f}). Wait {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                print(f" Early stopping: AP {val_ap:.3f} & Acc {val_acc:.3f} ≥ targets and no loss improvement for {self.patience} epochs.")
                return True
        return False


def run_final_test(model, opt):
    print("\n🔍 Running final evaluation on ForenSynths...")
    test_root = os.path.join(opt.root_dataset, 'ForenSynths')
    accs, aps = [], []

    for i, name in enumerate(vals):
        class_path = os.path.join(test_root, name)
        if not os.path.isdir(class_path):
            print(f" Skipping missing class: {name}")
            continue

        from copy import deepcopy
        opt_eval = deepcopy(opt)
        opt_eval.dataroot = class_path

        test_loader = create_dataloader(opt_eval)
        acc, ap = validate(model, test_loader)[:2]

        accs.append(acc)
        aps.append(ap)
        print(f"({i} {name:10}) acc: {acc * 100:.1f}%; ap: {ap * 100:.1f}%")

    print(f"({len(accs)} {'Mean':10}) acc: {np.mean(accs) * 100:.1f}%; ap: {np.mean(aps) * 100:.1f}%")
    print("*" * 40)



def main():
    opt = BaseOptions().parse()

    opt.root_dataset = opt.dataroot
    opt.dataroot = os.path.join(opt.root_dataset, opt.train_split)
    val_opt = opt
    val_opt.dataroot = os.path.join(opt.root_dataset, opt.val_split)

    data_loader = create_dataloader(opt)
    model = Trainer(opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    early_stopper = EarlyStopping(patience=10, target_ap=0.99)

    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(opt.niter):
        for i, data in enumerate(data_loader):
            model.total_steps += 1
            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(f"[{epoch:03d}] Step {model.total_steps}: loss = {model.loss:.4f}")
                train_writer.add_scalar('loss', model.loss, model.total_steps)

        if epoch % opt.delr_freq == 0 and epoch > 0:
            model.adjust_learning_rate()

        model.model.eval()
        acc_all, ap_all, val_loss_all, *_ = validate(model.model, val_opt)
        acc = float(np.mean(acc_all)) if isinstance(acc_all, list) else float(acc_all)
        ap = float(np.mean(ap_all)) if isinstance(ap_all, list) else float(ap_all)
        val_loss = float(np.mean(val_loss_all)) if isinstance(val_loss_all, list) else float(val_loss_all)

        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print(f"(Val @ epoch {epoch}) acc: {acc:.3f}; ap: {ap:.3f}; loss: {val_loss:.4f}")
        model.model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            model.save_networks(f"epoch_{epoch + 1}")
            model.save_networks("best_model")

        if early_stopper.check(val_loss, ap, acc):
            break

    model.save_networks("last")
    print(f"best model was from epoch {best_epoch + 1} with val loss = {best_val_loss:.4f}")
    model.model.eval()
    run_final_test(model.model, opt)


if __name__ == '__main__':
    main()
