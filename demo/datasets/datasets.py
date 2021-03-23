## -*- coding: utf-8 -*-
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT_DIR = os.path.abspath('/home/wtt/Documents/deep_rff_pytorch/demo')
sys.path.append(ROOT_DIR)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class ReshapeTransform:
    def __init__(self):
        pass

    def __call__(self, image):
        return image.view(-1)


class OneHot:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, label):
        y = torch.zeros(self.classes)
        y[label] = 1
        return y


# MNIST
def init_dataset(args):
    if args.dataset == 'MNIST':
        # train
        train_dataset = datasets.MNIST(root='data', train=True, download=True, \
                                       transform=transforms.Compose([
                                           transforms.ToTensor()
                                           # ReshapeTransform()
                                           # transforms.Normalize((0.1307,),(0.3081,))
                                       ]),
                                       target_transform=OneHot(classes=args.num_classes))

        # test
        test_dataset = datasets.MNIST(root='data', train=False, download=True, \
                                      transform=transforms.Compose([
                                          transforms.ToTensor()
                                          # ReshapeTransform()
                                          # transforms.Normalize((0.1307,),(0.3081,))
                                      ]),
                                      target_transform=OneHot(classes=args.num_classes))
    elif args.dataset == 'FMNIST':
        # train
        train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, \
                                              transform=transforms.Compose([
                                                  transforms.ToTensor()
                                                  # ReshapeTransform()
                                                  # transforms.Normalize((0.1307,),(0.3081,))
                                              ]),
                                              target_transform=OneHot(classes=args.num_classes))

        # test
        test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, \
                                             transform=transforms.Compose([
                                                 transforms.ToTensor()
                                                 # ReshapeTransform()
                                                 # transforms.Normalize((0.1307,),(0.3081,))
                                             ]),
                                             target_transform=OneHot(classes=args.num_classes))

    elif args.dataset == 'CIFAR10':
        # train
        train_dataset = datasets.CIFAR10(root='data', train=True, download=True, \
                                         transform=transforms.Compose([
                                             transforms.ToTensor()
                                             # ReshapeTransform()
                                             # transforms.Normalize((0.1307,),(0.3081,))
                                         ]),
                                         target_transform=OneHot(classes=args.num_classes))

        # test
        test_dataset = datasets.CIFAR10(root='data', train=False, download=True, \
                                        transform=transforms.Compose([
                                            transforms.ToTensor()
                                            # ReshapeTransform()
                                            # transforms.Normalize((0.1307,),(0.3081,))
                                        ]),
                                        target_transform=OneHot(classes=args.num_classes))
    elif args.dataset == 'CIFAR100':
        # train
        train_dataset = datasets.CIFAR100(root='data', train=True, download=True, \
                                         transform=transforms.Compose([
                                             transforms.ToTensor()
                                             # ReshapeTransform()
                                             # transforms.Normalize((0.1307,),(0.3081,))
                                         ]),
                                         target_transform=OneHot(classes=args.num_classes))

        # test
        test_dataset = datasets.CIFAR100(root='data', train=False, download=True, \
                                        transform=transforms.Compose([
                                            transforms.ToTensor()
                                            # ReshapeTransform()
                                            # transforms.Normalize((0.1307,),(0.3081,))
                                        ]),
                                        target_transform=OneHot(classes=args.num_classes))
    elif args.dataset == 'tiny-imagenet-200':
        train_path = '/home/wtt/Documents/deep_rff_pytorch/demo/datasets/tiny-imagenet-200/train/images'
        test_path = '/home/wtt/Documents/deep_rff_pytorch/demo/datasets/tiny-imagenet-200/val/images'

        train_dataset = datasets.ImageFolder(train_path, transform=transforms.ToTensor(),
                                             target_transform=OneHot(classes=args.num_classes))

        # print("*"*100)
        print(train_dataset.class_to_idx)
        test_dataset = datasets.ImageFolder(test_path, transform=transforms.ToTensor(),
                                             target_transform=OneHot(classes=args.num_classes))

    elif args.dataset == 'EuroSAT':
    # else:
        path = '/home/wtt/deep_rff_pytorch/demo/datasets/EuroSAT'
        train_dataset = datasets.ImageFolder(path, transform=transforms.ToTensor(),
                                             target_transform=OneHot(classes=args.num_classes))
        print(train_dataset.class_to_idx)
        test_dataset = None

    else:
        path = '/home/wtt/deep_rff_pytorch/demo/datasets/{}'.format(args.dataset)
        train_dataset = datasets.ImageFolder(path,
                                             transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((128,128))]),
                                             target_transform=OneHot(classes=args.num_classes))
        test_dataset = None

    if test_dataset is None:
        train_size = int(0.7 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

    val_size = int(args.validation_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    return train_loader, test_loader, val_loader

# dataset = datasets.ImageFolder('/home/wtt/deep_rff_pytorch/demo/datasets/EuroSAT', transform=transforms.ToTensor(),
#                                target_transform=OneHot(classes=10))
# train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
# for i, (data, target) in enumerate(train_loader):
#     print(data)
#     print(target)
# # print(dataset.classes)
# # print(dataset.class_to_idx)
# # print(dataset.imgs)
# img = dataset[0]
# print(dataset[0][0].shape)
# print(dataset[0])
