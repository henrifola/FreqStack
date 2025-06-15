import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

rz_dict = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
    'nearest': InterpolationMode.NEAREST
}

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.samples[index][0]
        return original_tuple + (path,)

def get_transforms(opt):
    rz_func = transforms.Lambda(lambda x: x) if opt.no_resize else transforms.Resize((opt.loadSize, opt.loadSize))
    crop_func = transforms.Lambda(lambda x: x) if opt.no_crop else (
        transforms.RandomCrop(opt.cropSize) if opt.isTrain else transforms.CenterCrop(opt.cropSize)
    )
    flip_func = transforms.Lambda(lambda x: x) if opt.no_flip or not opt.isTrain else transforms.RandomHorizontalFlip()

    return transforms.Compose([
        rz_func,
        crop_func,
        flip_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def dataset_folder(opt, root):
    return ImageFolderWithPaths(root=root, transform=get_transforms(opt))

def get_dataset(opt):
    classes = opt.classes if isinstance(opt.classes, list) else opt.classes.split()
    if not classes:
        classes = os.listdir(opt.dataroot)

    datasets = []
    for cls in classes:
        path = os.path.join(opt.dataroot, cls)
        if os.path.isdir(path):
            datasets.append(dataset_folder(opt, path))

    if not datasets:
        raise RuntimeError("No class folders found for: " + str(classes))

    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

def get_bal_sampler(dataset):
    targets = []
    if isinstance(dataset, ConcatDataset):
        for d in dataset.datasets:
            targets.extend(d.targets)
    else:
        targets = dataset.targets

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))

def create_dataloader(opt):
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None
    shuffle = False if opt.class_bal else not opt.serial_batches
    return DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle,
                      sampler=sampler, num_workers=opt.num_threads, drop_last=True)
