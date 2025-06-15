import os, time, torch, numpy as np
from sklearn.metrics import accuracy_score, average_precision_score

from options.base_options import BaseOptions
from network.trainer import Trainer
from validate import validate

# Base path resolution
LOALFREQ_ROOT = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(LOALFREQ_ROOT, '..'))

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


try:
    from util import printSet
except ImportError:
    def printSet(name: str):
        bar = "=" * 60
        print(f"\n{bar}\n Testing: {name}\n{bar}")


def main():
    opt = BaseOptions().parse()
    assert opt.model_path, "Provide --model_path"

    model = Trainer(opt).model
    model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
    model.cuda().eval()

    for test_name, cfg in DetectionTests.items():
        printSet(test_name)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        accs, aps = [], []
        dataroot = cfg['dataroot']
        include_only = cfg.get('include_only')

        for folder in sorted(os.listdir(dataroot)):
            if include_only and folder not in include_only:
                continue
            first_lvl = os.path.join(dataroot, folder)
            if not os.path.isdir(first_lvl):
                continue

            contents = set(os.listdir(first_lvl))
            leaf_paths = [first_lvl] if {'0_real', '1_fake'} <= contents else [
                os.path.join(first_lvl, sub)
                for sub in contents
                if os.path.isdir(os.path.join(first_lvl, sub)) and
                   {'0_real', '1_fake'} <= set(os.listdir(os.path.join(first_lvl, sub)))
            ]
            if not leaf_paths:
                continue

            all_preds, all_labels = [], []
            for leaf in leaf_paths:
                opt.dataroot = leaf
                opt.no_resize = cfg['no_resize']
                opt.no_crop = cfg['no_crop']
                opt.classes = ['']  # Dummy

                _, _, _, _, y_true, y_pred, *_ = validate(model, opt)
                all_preds.extend([float(p) for p in y_pred])
                all_labels.extend([float(l) for l in y_true])

            preds_bin = (np.asarray(all_preds) > 0.5).astype(float)
            acc_mean = accuracy_score(all_labels, preds_bin)
            ap_mean = average_precision_score(all_labels, all_preds)

            print(f"({folder:12}) acc: {acc_mean*100:.1f}; ap: {ap_mean*100:.1f}")
            accs.append(acc_mean)
            aps.append(ap_mean)

        if accs:
            mean_acc = np.mean(accs) * 100
            mean_ap = np.mean(aps) * 100
            print(f"({'Mean':12}) acc: {mean_acc:.1f}; ap: {mean_ap:.1f}")
        else:
            print("No valid folders found.")
        print("*" * 25)


if __name__ == "__main__":
    main()
