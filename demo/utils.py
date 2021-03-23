## -*- coding: utf-8 -*-
import torch


# FUNCTIONS
def KL_diagLog(mean, mean_prior, logsigma, logsigma_prior):
    logsigma_prior = logsigma_prior.view(-1, 1)
    # print(logsigma_prior.shape)
    A = logsigma_prior - logsigma
    B = torch.pow(mean - mean_prior, 2) / torch.exp(logsigma_prior)
    C = torch.exp(logsigma - logsigma_prior) - 1
    return 0.5 * torch.sum(A + B + C)


## Log-sum operation
def logsumexp(vals, dim=2):
    m = torch.max(vals, dim)[0]
    return m + torch.log(torch.sum(torch.exp(vals - m.unsqueeze(dim)), dim))
