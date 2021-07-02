import torch
from torch import nn


class cos_loss(nn.Module):
    def __init__(self):
        super(cos_loss,self).__init__()
    
    def forward(self,x,y):
        x = x.reshape(1,-1)
        y = y.reshape(1,-1)
        cos = nn.CosineSimilarity()
        loss = 1 - cos(x,y)
        return loss
