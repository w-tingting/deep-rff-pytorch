## -*- coding: utf-8 -*-
import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT_DIR = os.path.abspath('/home/wtt/deep_rff_pytorch/demo')
sys.path.append(ROOT_DIR)

import torch
import pickle
import argparse
from model import Net
from datasets import datasets
from likelihood import Softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deep rff')

    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='split fraction for validation  (default: 0.1)')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='number of label classes (default: 10)')
    parser.add_argument('--mc', type=int, default=5,
                        help='Monte Calro (default: 10)')
    parser.add_argument('--dataset', type=str, default="CIFAR10",
                        help='MNIST FMNIST CIFAR10 caltech4 EuroSAT (default: MNIST)')
    parser.add_argument('--kernel-type', type=str, default="RBF",
                        help='RBF arccos (default: RBF)')
    args = parser.parse_args()


    def idx_to_classes(args):
        if args.dataset is 'CIFAR10':
            info = open('./data/cifar-10-batches-py/batches.meta', 'rb')
            dict = pickle.load(info, encoding='bytes')
            id_to_cls = dict[b'label_names']
        elif args.dataset is 'EuroSAT':
            id_to_cls = {'AnnualCrop': 0, 'Forest': 1, 'HerbaceousVegetation': 2, 'Highway': 3, 'Industrial': 4,
                         'Pasture': 5, 'PermanentCrop': 6, 'Residential': 7, 'River': 8, 'SeaLake': 9}
        return id_to_cls


    train_loader, test_loader, _ = datasets.init_dataset(args)
    id_to_cls = idx_to_classes(args)
    model = Net(args.batch_size, args.mc,args.kernel_type,args.num_classes).to(device)
    model.load_state_dict(torch.load('CIFAR10_net_12-31-2020_17-53-54.pth'))

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            torch.cuda.synchronize()
            total_start = time.time()

            data = data.to(device)
            target = target.to(device)
            # prediction
            y_pred = model(data)
            prob = Softmax(args.batch_size, args.mc, args.num_classes).predict(y_pred).mean(0)
            y_pred = torch.argmax(prob, dim=1)
            confidence = torch.max(prob, dim=1)[0]
            classes = id_to_cls[y_pred]

            torch.cuda.synchronize()
            total_end = time.time()
            print("feed forward cost time: {:.4f}".format(total_end - total_start))
            target = torch.argmax(target, dim=1)
            print("*" * 100)
            print("y_pred: {} ground truth: {} classes: {} prob: {:.4f}\n".format(y_pred[0].cpu().numpy(),
                                                                                  target[0].cpu().numpy(),
                                                                                  str(classes, encoding='utf-8'),
                                                                                  confidence[0].cpu().numpy()))
            print('-' * 100)
