## -*- coding: utf-8 -*-
import torch
from torch import nn

from layers import Conv_RFF, Conv_Linear, FullyConnected
from likelihood import Softmax
from utils import KL_diagLog


class Net(nn.Module):
    def __init__(self, batch_size, mc, kernel_type, classes):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.mc = mc
        self.kernel_type = kernel_type
        self.classes = classes

        # in_channels, out_channels, k_size = 3, stride = 1, padding = 0, batch_size = 8, mc = 10, group, F0 = True/local_prame=True
        # MNIST
        self.rff0 = Conv_RFF(1, 16, 3, 2, 1, self.batch_size, self.mc, self.kernel_type, 1, True)
        self.linear0 = Conv_Linear(16 * 2, 16, 3, 2, 1, self.batch_size, self.mc, 1, False)

        self.rff1 = Conv_RFF(16, 16, 3, 1, 1, self.batch_size, self.mc, self.kernel_type, 1, False)
        self.linear1 = Conv_Linear(16 * 2, 16, 3, 2, 1, self.batch_size, self.mc, 1, False)

        self.rff2 = Conv_RFF(16, 16, 3, 2, 1, self.batch_size, self.mc, self.kernel_type, 1, False)
        self.fully = FullyConnected(2 * 2 * 16 * 2, self.classes, self.batch_size, self.mc, False)

       

    def forward(self, x):
        phi0 = self.rff0(x)
        F0 = self.linear0(phi0)

        phi1 = self.rff1(F0)
        F1 = self.linear1(phi1)

        x = self.rff2(F1)
        # print(x.shape)
        x = x.view(self.batch_size, self.mc, -1)
        x = x.transpose(0, 1)
        fully = self.fully(x)

        return fully

    def compute_objective(self, y_pred, y, num):
        ## Given the output layer, we compute the conditional likelihood across all samples
        softmax = Softmax(self.batch_size, self.mc, self.classes)
        ll = softmax.log_cond_prob(y, y_pred)
        ell = torch.sum(torch.mean(ll, 0)) * num
        return ell

    def get_kl(self):
        kl = 0
        for mod in self.modules():
            if isinstance(mod, Conv_RFF):
                # print(mod.__str__())
                Omega_mean_prior, Omega_logsigma_prior = mod.get_prior_Omega()
                kl += KL_diagLog(mod.Omega_mean, Omega_mean_prior, mod.Omega_logsigma,
                                 Omega_logsigma_prior)
            elif isinstance(mod, (Conv_Linear, FullyConnected)):
                # print(mod.__str__())
                # print(mod.W_eps[0])
                kl += KL_diagLog(mod.W_mean, mod.W_mean_prior, mod.W_logsigma,
                                 mod.W_logsigma_prior)
        return kl

    def freeze(self, iter=0):
        for child in self.children():
            if isinstance(child, Conv_RFF):
                if iter <= 10000:
                    child.theta_llscale.requires_grad = False
                    child.theta_logsigma.requires_grad = False
                    child.Omega_mean.requires_grad = False
                    child.Omega_logsigma.requires_grad = False
                    child.Omega_eps.requires_grad = False
                elif 10000 < iter <= 50000:
                    child.theta_llscale.requires_grad = False
                    child.theta_logsigma.requires_grad = False
                    child.Omega_mean.requires_grad = True
                    child.Omega_logsigma.requires_grad = True
                    child.Omega_eps.requires_grad = True
                else:
                    child.theta_llscale.requires_grad = True
                    child.theta_logsigma.requires_grad = True
                    child.Omega_mean.requires_grad = True
                    child.Omega_logsigma.requires_grad = True
                    child.Omega_eps.requires_grad = True
            elif isinstance(child, Conv_Linear):
                if iter <= 10000:
                    child.W_mean.requires_grad = True
                    child.W_logsigma.requires_grad = True
                    child.W_eps.requires_grad = True
                elif 10000 < iter <= 50000:
                    child.W_mean.requires_grad = False
                    child.W_logsigma.requires_grad = False
                    child.W_eps.requires_grad = False
                else:
                    child.W_mean.requires_grad = True
                    child.W_logsigma.requires_grad = True
                    child.W_eps.requires_grad = True
