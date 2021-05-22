import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, size_average=True, ignore_index=255, reduce=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(size_average=size_average, ignore_index=ignore_index,reduce=reduce)

    def forward(self, inputs, targets):
        log_p = F.log_softmax(inputs, dim=1)
        loss = self.nll_loss(log_p, targets)
        return loss


def to_one_hot_var(tensor, nClasses, requires_grad=False):

    n, h, w = tensor.size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.view(n, 1, h, w), 1)
    return Variable(one_hot, requires_grad=requires_grad)

class focal_loss(nn.Module):
    def __init__(self):
        super(focal_loss,self).__init__()
        self.sigmoid = nn.Sigmoid()    

    def forward(self,x,label):
        x = self.sigmoid(x)
        focal_loss = - label * (1-x)**2 * torch.log(x+1e-6) - (1-label) * x**2 * torch.log(1-x+1e-6)
        return torch.sum(focal_loss) / x.nelement()


def to_one_hot(tensor,nClasses):
    
    n,h,w = tensor.size()
    one_hot = torch.zeros(n,nClasses,h,w).scatter_(1,tensor.view(n,1,h,w),1)
    return one_hot

class mIoULoss(nn.Module):
    def __init__(self, n_classes,device):
        super(mIoULoss, self).__init__()
        self.n_classes = n_classes
        self.device = device

    def to_one_hot(self,tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).to(self.device).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)
        
        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()