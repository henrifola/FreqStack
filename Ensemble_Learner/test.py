import sys
import os
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
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

DetectionTests = {
    'Trained classes': {
        'dataroot': rel_path('FreqNet_DeepfakeDetection', 'dataset', 'EnsembleTest', 'progan'),
        'include_only': ['car', 'chair', 'horse', 'cat'],
        'no_resize': False,
        'no_crop': True,
    },
    'Perturbed trained classes': {
        'dataroot': rel_path('FreqNet_DeepfakeDetection', 'dataset', 'perturbed-data', 'test', 'ForenSynths', 'progan'),
        'include_only': ['car', 'chair', 'horse', 'cat'],
        'no_resize': False,
        'no_crop': True,
    },
    'Original Data': {
        'dataroot': rel_path('FreqNet_DeepfakeDetection', 'dataset', 'ForenSynths'),
        'no_resize': False,
        'no_crop': True,
        'include_only': [
            'progan', 'stylegan', 'stylegan2', 'biggan',
            'cyclegan', 'stargan', 'gaugan', 'deepfake'
        ]
    },
    'Perturbed-Subset': {
        'dataroot': rel_path('FreqNet_DeepfakeDetection', 'dataset', 'perturbed-data', 'test', 'ForenSynths'),
        'no_resize': False,
        'no_crop': True,
        'include_only': [
            'progan', 'stylegan', 'stylegan2', 'biggan',
            'cyclegan', 'stargan', 'gaugan', 'deepfake'
        ]
    },
    'Blur-50%': {
        'dataroot': rel_path('FreqNet_DeepfakeDetection', 'dataset', 'perturbed-data', 'test', 'blur-50%'),
        'no_resize': False,
        'no_crop': True,
        'include_only': [
            'progan', 'stylegan', 'stylegan2', 'biggan',
            'cyclegan', 'stargan', 'gaugan', 'deepfake'
        ]
    },
    'Noise-50%': {
        'dataroot': rel_path('FreqNet_DeepfakeDetection', 'dataset', 'perturbed-data', 'test', 'noise-50%'),
        'no_resize': False,
        'no_crop': True,
        'include_only': [
            'progan', 'stylegan', 'stylegan2', 'biggan',
            'cyclegan', 'stargan', 'gaugan', 'deepfake'
        ]
    },
    'Compress-50%': {
        'dataroot': rel_path('FreqNet_DeepfakeDetection', 'dataset', 'perturbed-data', 'test', 'compress-50%'),
        'no_resize': False,
        'no_crop': True,
        'include_only': [
            'progan', 'stylegan', 'stylegan2', 'biggan',
            'cyclegan', 'stargan', 'gaugan', 'deepfake'
        ]
    },
}


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


def printSet(name):
    bar = '=' * 60
    print(f"\n{bar}\nTesting MetaLearner: {name}\n{bar}")


def run_test():
    freq_opt = FreqNetOptions().parse(print_options=False)
    ll_opt = LLNetOptions().parse(print_options=False)

    model1 = load_branch1(freq_opt)
    model2 = load_branch2(ll_opt)
    

    metalearner = MetaLearner().cuda()
    metalearner_path = os.path.join(ENSEMBLE_ROOT, 'checkpoints_metalearner', '20250613_063609', 'last_model.pth')
    metalearner.load_state_dict(torch.load(metalearner_path))
    metalearner.eval()

    for test_name, cfg in DetectionTests.items():
        printSet(test_name)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        dataroot = cfg['dataroot']
        include_only = cfg.get('include_only')
        accs, aps = [], []
        v_id = 0

        for folder in sorted(os.listdir(dataroot)):
            if include_only and folder not in include_only:
                continue
            first_lvl = os.path.join(dataroot, folder)
            if not os.path.isdir(first_lvl):
                continue

            contents = set(os.listdir(first_lvl))
            if '0_real' in contents and '1_fake' in contents:
                leaf_paths = [first_lvl]
            else:
                leaf_paths = [
                    os.path.join(first_lvl, sub)
                    for sub in contents
                    if os.path.isdir(os.path.join(first_lvl, sub))
                    and '0_real' in os.listdir(os.path.join(first_lvl, sub))
                    and '1_fake' in os.listdir(os.path.join(first_lvl, sub))
                ]
            if not leaf_paths:
                continue

            folder_preds, folder_labels = [], []

            for leaf in leaf_paths:
                for opts in (freq_opt, ll_opt):
                    opts.dataroot = leaf
                    opts.no_resize = cfg['no_resize']
                    opts.no_crop = cfg['no_crop']

                _, _, _, _, y_true_1, y_pred_1, paths_1 = validate_branch1.validate(model1, freq_opt)
                _, _, _, _, y_true_2, y_pred_2, paths_2 = validate_branch2.validate(model2, ll_opt)

                assert set(paths_1) == set(paths_2), "Mismatch in evaluated image paths between branches"

                global_dict = {p: (pred, lbl) for p, pred, lbl in zip(paths_1, y_pred_1, y_true_1)}
                local_dict = {p: pred for p, pred in zip(paths_2, y_pred_2)}
                common = list(set(global_dict) & set(local_dict))
                if not common:
                    continue

                X, Y = [], []
                for p in common:
                    p1, lbl = global_dict[p]
                    p2 = local_dict[p]
                    p1 = float(p1.squeeze()) if isinstance(p1, (np.ndarray, torch.Tensor)) else float(p1)
                    p2 = float(p2.squeeze()) if isinstance(p2, (np.ndarray, torch.Tensor)) else float(p2)
                    X.append([p1, p2])
                    Y.append(float(lbl))

                X = torch.from_numpy(np.asarray(X, dtype=np.float32)).cuda()
                Y = torch.from_numpy(np.asarray(Y, dtype=np.float32)).cuda()

                with torch.no_grad():
                    preds = metalearner(X).squeeze().cpu().numpy()
                folder_preds.extend(preds)
                folder_labels.extend(Y.cpu().numpy())

            if folder_preds:
                preds_bin = (np.asarray(folder_preds) > 0.5).astype(float)
                acc = accuracy_score(folder_labels, preds_bin)
                ap = average_precision_score(folder_labels, folder_preds)
                print(f"({v_id} {folder:12}) acc: {acc*100:.1f}; ap: {ap*100:.1f}")
                accs.append(acc)
                aps.append(ap)
                v_id += 1

        if accs:
            mean_acc = np.mean(accs) * 100
            mean_ap = np.mean(aps) * 100
            print(f"({v_id} {'Mean':12}) acc: {mean_acc:.1f}; ap: {mean_ap:.1f}")
        else:
            print("(0 Mean        ) acc: 0.0; ap: 0.0")

        print('*' * 25)


if __name__ == "__main__":
    run_test()
