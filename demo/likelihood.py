## -*- coding: utf-8 -*-
import torch

from utils import logsumexp


class Softmax:
    """
    Implements softmax likelihood for multi-class classification
    """

    def __init__(self,batch_size,mc,classes):
        self.batch_size = batch_size
        self.mc = mc
        self.classes = classes

    def log_cond_prob(self, y, y_pred):
        a =torch.sum(y * y_pred, 2)
        b =logsumexp(y_pred, 2)
        # print("y_pred",torch.sum(y_pred))
        prob = a-b
        # print("loss",prob)
        return prob

    def predict(self, y_pred):
        """
        return the probabilty for all the samples, datapoints and classes
        param: y_pred
        return:
        """
        logprob = y_pred - logsumexp(y_pred, 2).unsqueeze(2).expand(self.mc, self.batch_size, self.classes)
        return torch.exp(logprob)
