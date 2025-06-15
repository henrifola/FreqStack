import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from metalearner import MetaLearner

# Base path resolution
ENSEMBLE_ROOT = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ENSEMBLE_ROOT, '..'))

sys.path.append(os.path.join(PROJECT_ROOT, 'FreqNet_DeepfakeDetection'))
sys.path.append(os.path.join(PROJECT_ROOT, 'LoalFreq'))
sys.path.append(PROJECT_ROOT)

from FreqNet_DeepfakeDetection.options.test_options import TestOptions as FreqNetOptions
from FreqNet_DeepfakeDetection.networks.freqnet import freqnet
import FreqNet_DeepfakeDetection.validate as validate_branch1

from LoalFreq.options.base_options import BaseOptions as LLNetOptions
from LoalFreq.network.trainer import Trainer
import LoalFreq.validate as validate_branch2

def rel_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

def load_branch1(opt):
    model = freqnet(num_classes=1)
    state_dict = torch.load(
        rel_path('FreqNet_DeepfakeDetection', 'checkpoints', 'experiment_name2025_05_29_04_13_40', 'model_epoch_last.pth'),
        map_location='cpu'
    )
    model.load_state_dict(state_dict)
    model.cuda().eval()
    return model

def load_branch2(opt):
    trainer = Trainer(opt)
    checkpoint = torch.load(
        rel_path('LoalFreq', 'checkpoints', '4class-llnet-car-cat-chair-horse-sgd_20250604_063140', 'epoch_60_model.pth'),
        map_location='cpu'
    )
    trainer.model.load_state_dict(checkpoint)
    model = trainer.model
    model.cuda().eval()
    return model

def main():
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(ENSEMBLE_ROOT, 'checkpoints_metalearner', timestamp)
    os.makedirs(save_dir, exist_ok=True)

    freq_opt = FreqNetOptions().parse(print_options=False)
    freq_opt.dataroot = rel_path('FreqNet_DeepfakeDetection', 'dataset', 'EnsembleTrain', 'progan')
    freq_opt.classes = ['car', 'cat', 'chair', 'horse']
    freq_opt.no_resize = False
    freq_opt.no_crop = True

    ll_opt = LLNetOptions().parse(print_options=False)
    ll_opt.dataroot = rel_path('FreqNet_DeepfakeDetection', 'dataset', 'EnsembleTrain', 'progan')
    ll_opt.classes = ['car', 'cat', 'chair', 'horse']
    ll_opt.wavelet_type = 'db1'
    ll_opt.window_size = 32

    model1 = load_branch1(freq_opt)
    model2 = load_branch2(ll_opt)

    _, _, _, _, y_true_1, y_pred_1, paths_1 = validate_branch1.validate(model1, freq_opt)
    _, _, y_pred_2, y_true_2, _, paths_2 = validate_branch2.validate(model2, ll_opt)

    global_dict = {p: (pred, label) for p, pred, label in zip(paths_1, y_pred_1, y_true_1)}
    local_dict = {p: pred for p, pred in zip(paths_2, y_pred_2)}

    common_paths = list(set(global_dict.keys()) & set(local_dict.keys()))
    X, Y = [], []

    for path in common_paths:
        p1, label = global_dict[path]
        p2 = local_dict[path]
        X.append([p1, p2])
        Y.append(label)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MetaLearner().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(80):
        model.train()
        epoch_loss = 0
        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.cuda(), yb.cuda()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"[{epoch:03d}] Sample {i:03d}: loss = {loss.item():.4f}")
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            preds = model(X.cuda()).squeeze().cpu().numpy()
            val_loss = loss_fn(model(X.cuda()).squeeze(), Y.cuda()).item()
            acc = accuracy_score(Y, preds > 0.5)
            print(f"(Val @ epoch {epoch:03d}) acc: {acc:.3f}; loss: {val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1:03d}_model.pth"))

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_{best_epoch:03d}.pth"))

        if acc >= 0.99:
            print(f"Early stopping triggered at epoch {epoch+1:03d} (accuracy = {acc:.3f} ≥ 0.99)")
            break

    torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))
    print(f"Meta-learner trained. Best val loss = {best_loss:.4f} at epoch {best_epoch:03d}. Models saved to {save_dir}")


if __name__ == "__main__":
    main()
