import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from data import create_dataloader
from options.base_options import BaseOptions
from network.LLnet import LLnet


def validate(model, opt):
    dataloader = create_dataloader(opt)

    y_true, y_pred, paths = [], [], []

    with torch.no_grad():
        for inputs, labels, image_paths in dataloader:
            inputs = inputs.cuda()
            outputs = model(inputs).sigmoid().flatten()

            y_pred.extend(outputs.cpu().tolist())
            y_true.extend(labels.flatten().cpu().tolist())
            paths.extend(image_paths)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] < 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] >= 0.5)
    acc = accuracy_score(y_true, y_pred >= 0.5)
    ap = average_precision_score(y_true, y_pred)

    return acc, ap, r_acc, f_acc, y_true, y_pred, paths


if __name__ == '__main__':
    opt = BaseOptions().parse(print_options=False)

    model = LLnet(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred, paths = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)