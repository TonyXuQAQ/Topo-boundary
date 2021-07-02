import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from .spatial_argmax import *


class DecoderCoord(nn.Module):
    def __init__(self, d_model=256,visual_size=29*29):
        super().__init__()
        self.conv1 = nn.Conv2d(9, 8, 3, stride=1, padding=1)  # b, 8, 101, 101
        self.pool1 = nn.MaxPool2d(2, stride=2)  # b, 8, 50,50
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=0)  # b, 8, 48,48
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, 1, stride=1, padding=0)  # b, 1, 24,24
        self.input_linear = nn.Linear( visual_size, d_model)
        self.output_1 = nn.Linear(d_model, d_model // 2)
        self.output_2 = nn.Linear(d_model // 2, 2)
    
    def forward(self, x, y, z):
        x = F.leaky_relu_(self.conv1(x))
        x = self.conv1_bn(self.pool1(x))
        x = self.conv2_bn(F.leaky_relu_(self.conv2(x)))
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        z = z.reshape(z.shape[0], -1)
        x = torch.cat([x,y,z],dim=1)
        output = F.relu_(self.input_linear(x))
        output = F.relu(self.output_1(output))
        output = self.output_2(output)
        return output

class DecoderStop(nn.Module):
    def __init__(self, d_model=256,visual_size=61*61):
        super().__init__()
        self.conv1 = nn.Conv2d(9, 8, 3, stride=1, padding=1)  # b, 8, 63,63
        self.pool1 = nn.MaxPool2d(2, stride=2)  # b, 8, 31,31
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=0)  # b, 16, 29,29
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, 1, stride=1, padding=0)  # b, 1, 29,29
        self.input_linear = nn.Linear( visual_size, d_model)
        self.output_1 = nn.Linear(d_model, d_model // 2)
        self.output_2 = nn.Linear(d_model // 2, 2)
    
    def forward(self, x, y, z):
        x = F.leaky_relu_(self.conv1(x))
        x = self.conv1_bn(self.pool1(x))
        x = self.conv2_bn(F.leaky_relu_(self.conv2(x)))
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        z = z.reshape(z.shape[0], -1)
        # print(x.shape,y.shape,z.shape)
        x = torch.cat([x,y,z],dim=1)
        output = F.relu_(self.input_linear(x))
        output = F.relu(self.output_1(output))
        output = self.output_2(output)
        return output

class FPN_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward(self,x):
        return self._upsample(self.layer1(x), 1000,1000)