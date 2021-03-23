## -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


# theta+Omega
class Conv_RFF(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=0, batch_size=8, mc=10, kernel_type='RBF',
                 group=1, F0=True):
        super(Conv_RFF, self).__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels // group
        self.kernel_size = k_size
        self.stride = stride
        self.padding = padding
        self.batch_size = batch_size
        self.mc = mc
        self.kernel_type = kernel_type
        self.group = group
        self.F0 = F0

        # theta:scaler
        self.theta_logsigma = nn.Parameter(torch.zeros(1), requires_grad=True)
        # scaler
        self.llscale = nn.Parameter(0.5 * torch.log(torch.tensor(self.in_channels * k_size * k_size).float()),
                                    requires_grad=False)
        # [C*k*k,C']
        self.theta_llscale = nn.Parameter(
            torch.ones([self.in_channels * k_size * k_size]) * self.llscale, requires_grad=True)
        # Omega:q
        # Omega:[C*k*k,C']
        self.Omega_mean = nn.Parameter(torch.zeros(self.in_channels * k_size * k_size, self.out_channels),
                                       requires_grad=True)
        # Omega:[C*k*k,C']
        self.Omega_logsigma = nn.Parameter(
            (self.theta_llscale * -2.)[..., None] * torch.ones(self.in_channels * k_size * k_size, self.out_channels),
            requires_grad=True)
        # Omega_eps:[mc,C*k*k,C']
        self.Omega_eps = nn.Parameter(
            torch.randn(self.mc, self.in_channels * self.kernel_size * self.kernel_size, self.out_channels),
            requires_grad=True)

    def get_prior_Omega(self):
        # [C*k*k,1]
        Omega_mean_prior = torch.zeros(self.in_channels * self.kernel_size * self.kernel_size, 1).cuda()
        # [C*k*k]
        Omega_logsigma_prior = self.theta_llscale * -2. * torch.ones(
            self.in_channels * self.kernel_size * self.kernel_size).cuda()
        return Omega_mean_prior, Omega_logsigma_prior

    def forward(self, x):
        # input:
        # original image:[batch_size,C,H,W]->[8,1,28,28]
        # or the output feature map of last layer:[batch_size,mc*channels,H,W]
        if self.F0:
            x = torch.unsqueeze(x, 0).repeat(self.mc, 1, 1, 1, 1)
            H = W = x.size(-1)
            x = x.transpose(0, 1)
            x = x.reshape(self.batch_size, self.mc * self.in_channels, H, W)
        # Omega_eps = nn.Parameter(
        #     torch.randn(self.mc, self.in_channels * self.kernel_size * self.kernel_size, self.out_channels)).cuda()

        # local reparameter
        # self.Omega_from_q = Omega_eps * torch.exp(self.Omega_logsigma / 2) + self.Omega_mean
        self.Omega_from_q = self.Omega_eps * torch.exp(self.Omega_logsigma / 2) + self.Omega_mean

        # conv2d_rff
        # x:[batch_size,mc*channels,H,W]
        # weight:self.Omega_from_q [mc,C*k*k,C']
        # output:[batch,mc*channels,H,W]
        phi_half = conv2d(x, self.Omega_from_q, self.in_channels, self.out_channels, self.kernel_size, self.stride,
                          self.padding, self.mc, self.group)
        ## RFF:[cos(F*x),sin(F*x)]
        phi_half_size = phi_half.shape[-1]
        N_rf = self.out_channels * phi_half_size * phi_half_size
        phi = torch.exp(0.5 * self.theta_logsigma)
        phi = phi / torch.sqrt(torch.tensor(1.) * N_rf)

        phi_half = phi_half.view(self.batch_size, self.mc, self.out_channels, phi_half_size, phi_half_size)
        if self.kernel_type == "RBF":
            A = phi * torch.cos(phi_half)
            B = phi * torch.sin(phi_half)
            # output:[batch,mc*channels*2,H,W]
            phi = torch.cat([A, B], 2)
        else:
            phi = phi * torch.cat([torch.maximum(phi_half, torch.tensor(0.).cuda())], 2)
        phi = phi.view(self.batch_size, self.mc * phi.shape[2], phi_half_size, phi_half_size)

        return phi


# W
class Conv_Linear(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=0, batch_size=8, mc=10, group=1,
                 local_reparam=True, F0=False):
        super(Conv_Linear, self).__init__()

        self.in_channels = in_channels // group
        self.out_channels = out_channels
        self.kernel_size = k_size
        self.stride = stride
        self.padding = padding
        self.batch_size = batch_size
        self.mc = mc
        self.group = group
        self.local_reparam = local_reparam
        self.F0 = F0

        # scaler
        self.W_mean_prior = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.W_mean = nn.Parameter(
            torch.ones(self.in_channels * self.kernel_size * self.kernel_size, self.out_channels), requires_grad=True)
        # scaler
        self.W_logsigma_prior = nn.Parameter(torch.zeros(1), requires_grad=False)
        # [C*k*k,C']
        self.W_logsigma = nn.Parameter(
            torch.ones(self.in_channels * self.kernel_size * self.kernel_size, self.out_channels), requires_grad=True)

        if local_reparam:
            self.register_parameter('W_eps', None)
        else:
            # W:[C*k*k,C']
            self.W_eps = nn.Parameter(
                torch.randn(self.mc, self.in_channels * self.kernel_size * self.kernel_size, self.out_channels),
                requires_grad=True)

    def reset_parameters(self, input):
        self.W_eps = nn.Parameter(input.new(input.size()).normal_(0, 1), requires_grad=True)

    def forward(self, phi):
        if self.F0:
            phi = torch.unsqueeze(phi, 0).repeat(self.mc, 1, 1, 1, 1)
            H = W = phi.size(-1)
            phi = phi.transpose(0, 1)
            phi = phi.reshape(self.batch_size, self.mc * self.in_channels, H, W)

        # local reparam
        if self.local_reparam:
            # [mc,C*k*k,C']
            # W_mean = torch.unsqueeze(self.W_mean, 0).repeat(self.mc, 1, 1)
            # W_logsigma = torch.unsqueeze(self.W_logsigma, 0).repeat(self.mc, 1, 1)
            W_mean = self.W_mean.permute(1, 0)
            W_logsigma = self.W_logsigma.permute(1, 0)

            W_mean_F = F.conv2d(phi, W_mean, stride=self.stride, padding=self.padding)
            W_logsigma_F = F.conv2d(torch.pow(phi, 2), torch.exp(W_logsigma), stride=self.stride, padding=self.padding)

            if self.W_eps is None:
                self.reset_parameters(W_logsigma_F)
            F_y = W_mean_F + torch.sqrt(W_logsigma_F) * self.W_eps

        else:
            W_from_q = (self.W_mean + torch.exp(self.W_logsigma / 2.) * self.W_eps)
            ## conv2d
            F_y = conv2d(phi, W_from_q, self.in_channels, self.out_channels, self.kernel_size, self.stride,
                         self.padding, self.mc, self.group)
        return F_y


# W
class FullyConnected(nn.Module):
    def __init__(self, in_features, out, batch_size=8, mc=10, local_reparam=False):
        super(FullyConnected, self).__init__()
        self.in_features = in_features
        self.out = out
        self.batch_size = batch_size
        self.mc = mc
        self.local_reparam = local_reparam
        # [C*H*W,C']
        self.W_mean_prior = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.W_mean = nn.Parameter(torch.ones(self.in_features, self.out), requires_grad=True)
        self.W_logsigma_prior = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.W_logsigma = nn.Parameter(torch.ones(self.in_features, self.out), requires_grad=True)

        if local_reparam:
            # [mc,batch_size,C']
            # self.W_eps = nn.Parameter(torch.randn(self.mc, self.batch_size, self.out), requires_grad=True)
            self.W_eps = nn.Parameter(torch.randn(self.mc, 1, self.out), requires_grad=True)
        else:
            # [mc,C*H*W,C']
            self.W_eps = nn.Parameter(torch.randn(self.mc, self.in_features, self.out), requires_grad=True)

    def forward(self, phi):
        # print("phi",phi)
        # local reparam
        if self.local_reparam:

            # W_eps = nn.Parameter(torch.randn(self.mc, 1, self.out)).cuda()
            # [mc,C*H*W,C']
            W_mean = torch.unsqueeze(self.W_mean, 0).repeat(self.mc, 1, 1)
            W_logsigma = torch.unsqueeze(self.W_logsigma, 0).repeat(self.mc, 1, 1)
            # phi:[mc,batch_size,C*H*W]
            # W_mean:[mc,C*H*W,C']
            W_mean_F = torch.matmul(phi, W_mean)
            W_logsigma_F = torch.matmul(torch.pow(phi, 2), torch.exp(W_logsigma))
            # F_y:[mc,batch_size,C']
            F_y = W_mean_F + torch.sqrt(W_logsigma_F) * self.W_eps
        else:
            # [mc,C*H*W,C']
            W_from_q = (self.W_mean + torch.exp(self.W_logsigma / 2.) * self.W_eps)
            # phi:[mc,batch_size,C*H*W]
            # W_from_q:[mc,C*H*W,C']
            # F_y:[mc,batch_size,C']

            # print(phi.shape)
            # print(W_from_q.shape)
            F_y = torch.matmul(phi, W_from_q)
        return F_y


class Resnet(nn.Module):
    """Implement for Resnet"""

    def __init__(self, batch_size, mc):
        super(Resnet, self).__init__()
        self.batch_size = batch_size
        self.mc = mc

    def forward(self, x, F):
        feature_size = F.size(-1)
        x = nn.AdaptiveAvgPool2d(feature_size)(x)
        x = x.view(self.batch_size, self.mc, -1, feature_size, feature_size)
        F = F.view(self.batch_size, self.mc, -1, feature_size, feature_size)
        F_y = torch.cat([x, F], 2)
        # F_y = torch.transpose(F_y, 1, 2).contiguous()
        # F_y = torch.add(x, F)
        F_y = F_y.view(self.batch_size, -1, feature_size, feature_size)
        return F_y


class Conv_T(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, out_padding, mc):
        super(Conv_T, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_padding = out_padding
        self.mc = mc

    def forward(self, x):
        up = nn.Upsample(scale_factor=2, mode='nearest').cuda()(x)
        # dconv = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels,
        #                            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
        #                            output_padding=self.out_padding, groups=self.mc,
        #                            bias=False).cuda()(x)
        return up


def conv2d(x, weight, in_channel, out_channel, k_size=3, stride=2, padding=0, mc=10, group=1):
    weight = weight.view(mc, k_size, k_size, in_channel, out_channel)
    weight = weight.permute(0, 4, 3, 1, 2)
    weight = weight.reshape(out_channel * mc, int(in_channel), k_size, k_size)

    result = F.conv2d(x, weight, stride=stride, groups=mc * group, padding=padding)
    return result
