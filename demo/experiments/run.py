## -*- coding: utf-8 -*-
import os
import sys
import time
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT_DIR = os.path.abspath('/home/wtt/Documents/deep_rff_pytorch/demo')
sys.path.append(ROOT_DIR)

import torch
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = True

import argparse
from utils import logsumexp
from model import Net
from datasets import datasets
from torch.optim import lr_scheduler
from likelihood import Softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deep rff')

    parser.add_argument('--random-seed', type=int, default=123456,
                        help='random seed(default: 123456)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='split fraction for validation  (default: 0.1)')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='number of label classes (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Initial learning rate. (default: 0.01)')
    parser.add_argument('--display-step', type=float, default=1000,
                        help='Display progress every FLAGS.display_step iterations (default: 2000)')
    parser.add_argument('--mc', type=int, default=50,
                        help='Monte Calro (default: 10)')
    parser.add_argument('--kernel-type', type=str, default="RBF",
                        help='RBF arccos (default: RBF)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='train epoches (default: 50)')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='MNIST FMNIST CIFAR10 CIFAR100 caltech4 tiny-imagenet-200(default: MNIST)')
    args = parser.parse_args()

    train_loader, test_loader, _ = datasets.init_dataset(args)
    model = Net(args.batch_size, args.mc, args.kernel_type, args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=49, eta_min=5e-5, last_epoch=-1)

    lr = []
    iter = 0
    torch.cuda.synchronize()
    total_start = time.time()

    for ep in range(args.epochs):
        torch.cuda.synchronize()
        start = time.time()

        print("Epoch {}/{}".format(ep + 1, args.epochs))
        print("-" * 50)

        running_loss = 0.0
        running_kl = 0.0
        running_nell = 0.0

        model.train()
        for i, (data, target) in enumerate(train_loader):

            iter += 1
            # model.freeze(iter)
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            y_pred = model(data)
            ell = model.compute_objective(y_pred, target, len(train_loader))
            kl = model.get_kl()
            loss = kl - ell
            loss.backward()
            # print("grad",model.linear1.W_logsigma.grad)
            optimizer.step()
            running_loss += loss.item()
            running_kl += kl
            running_nell += -ell

            if i % args.display_step == 99:
                # validation
                error_rate = 0.000
                nlpp = 0.000
                total = 0

                model.eval()
                with torch.no_grad():
                    for data, target in test_loader:
                        data = data.to(device)
                        target = target.to(device)

                        y_pred = model(data)
                        # print(y_pred.shape)
                        softmax = Softmax(args.batch_size, args.mc, args.num_classes)
                        nlpp += -torch.sum(-torch.log(torch.tensor(args.mc).float()) + logsumexp(
                            (softmax.log_cond_prob(target, y_pred)), 0))

                        y_pred_prob = softmax.predict(y_pred)
                        y_pred = y_pred_prob.mean(0)
                        y_pred = torch.argmax(y_pred, dim=1)
                        target = torch.argmax(target, dim=1)

                        total += target.size(0)
                        error_rate += (y_pred != target).sum()

                    print(
                        "Epoch [{}, {}] learning rate: {:.6f} loss: {:.4f} kl: {:.2f} nell: {:.2f} error_rate: {:.3f}% nlpp: {:.3f}\n".format(
                            ep + 1, i + 1, optimizer.state_dict()['param_groups'][0]['lr'],
                            running_loss / args.display_step, running_kl / args.display_step,
                            running_nell / args.display_step, 100.000 * error_rate.float() / total, nlpp.float() / total))

                    running_loss = 0.0
                    running_kl = 0.0
                    running_nell = 0.0

        scheduler.step()

        torch.cuda.synchronize()
        end = time.time()
        print("-" * 100)
        print("Epoch {} cost time: {:.4f}".format(ep + 1, (end - start)))
        lr.append(optimizer.state_dict()['param_groups'][0]['lr'])

    torch.cuda.synchronize()
    total_end = time.time()
    print("-" * 100)
    print("All Epoch cost time: {:.4f}".format(total_end - total_start))

    current_time = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    PATH = '{}_net_{}.pth'.format(args.dataset, current_time)
    torch.save(model.state_dict(), PATH)

    # import matplotlib.pyplot as plt
    # plt.plot(range(50), lr, color='r', label='lr')
    # plt.legend()
    # plt.show()
