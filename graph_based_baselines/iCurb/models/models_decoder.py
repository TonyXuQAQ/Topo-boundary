import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=3,stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels  , kernel_size=kernel_size, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels  )
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels  , kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels  )
            )
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class DecoderCoord(nn.Module):
    def __init__(self,visual_size):
        super().__init__()
        self.conv = nn.Conv2d(9,16,1,1)
        self.res_layer1 = BasicBlock(16,16,stride=1)
        self.res_layer2 = BasicBlock(16,16,stride=1)
        self.res_layer3 = BasicBlock(16,32,stride=2)
        self.res_layer4 = BasicBlock(32,32,stride=2)
        self.max_pool = nn.MaxPool2d(2,stride=2)
        self.linear1 = nn.Linear(visual_size,visual_size//2)
        self.linear2 = nn.Linear(visual_size//2,2)

    def forward(self,x,y,z):
        x1 = self.conv(x)
        x2 = self.res_layer4(self.res_layer3(self.res_layer2(self.res_layer1(x1))))
        x3 = self.max_pool(x2)
        
        x3 = x3.reshape(x3.shape[0],-1)
        y = y.reshape(y.shape[0], -1)
        z = z.reshape(z.shape[0], -1)
        x4 = torch.cat([x3,y,z],dim=1)
        
        x5 = nn.ReLU(inplace=True)(self.linear1(x4))
        x6 = self.linear2(x5)
        return x6

class DecoderStop(nn.Module):
    def __init__(self,visual_size):
        super().__init__()
        self.conv = nn.Conv2d(9,16,1,1)
        self.res_layer1 = BasicBlock(16,16,stride=1)
        self.res_layer2 = BasicBlock(16,16,stride=1)
        self.res_layer3 = BasicBlock(16,32,stride=2)
        self.res_layer4 = BasicBlock(32,32,stride=2)
        self.max_pool = nn.MaxPool2d(2,stride=2)
        self.linear1 = nn.Linear(visual_size,visual_size//2)
        self.linear2 = nn.Linear(visual_size//2,2)

    def forward(self,x,y,z):
        x1 = self.conv(x)
        x2 = self.res_layer4(self.res_layer3(self.res_layer2(self.res_layer1(x1))))
        x3 = self.max_pool(x2)
        x3 = x3.reshape(x3.shape[0],-1)
        y = y.reshape(y.shape[0], -1)
        z = z.reshape(z.shape[0], -1)
        x4 = torch.cat([x3,y,z],dim=1)
        x5 = nn.ReLU(inplace=True)(self.linear1(x4))
        x6 = self.linear2(x5)
        return x6

# class DecoderCoord(nn.Module):
#     def __init__(self, d_model=256,visual_size=29*29):
#         super().__init__()
#         self.conv1 = nn.Conv2d(9, 16, 3, stride=1, padding=1)  # b, 8, 101, 101
#         self.pool1 = nn.MaxPool2d(2, stride=2)  # b, 8, 50,50
#         self.conv1_bn = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=0)  # b, 8, 48,48
#         self.conv2_bn = nn.BatchNorm2d(8)
#         self.conv3 = nn.Conv2d(8, 1, 1, stride=1, padding=0)  # b, 1, 24,24
#         self.input_linear = nn.Linear( visual_size, d_model)
#         self.output_1 = nn.Linear(d_model, d_model // 2)
#         self.output_2 = nn.Linear(d_model // 2, 2)
    
#     def forward(self, x, y, z):
#         x = F.relu(self.conv1(x))
#         x = self.conv1_bn(self.pool1(x))
#         x = self.conv2_bn(F.relu(self.conv2(x)))
#         x = self.conv3(x)
#         x = x.reshape(x.shape[0], -1)
#         y = y.reshape(y.shape[0], -1)
#         z = z.reshape(z.shape[0], -1)
#         x = torch.cat([x,y,z],dim=1)
#         output = F.relu(self.input_linear(x))
#         output = F.relu(self.output_1(output))
#         output = self.output_2(output)
#         return output

# class DecoderStop(nn.Module):
#     def __init__(self, d_model=256,visual_size=61*61):
#         super().__init__()
#         self.conv1 = nn.Conv2d(9, 16, 3, stride=1, padding=1) 
#         self.pool1 = nn.MaxPool2d(2, stride=2)  
#         self.conv1_bn = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=0)  
#         self.conv2_bn = nn.BatchNorm2d(8)
#         self.conv3 = nn.Conv2d(8, 1, 1, stride=1, padding=0)  
#         self.input_linear = nn.Linear( visual_size, d_model)
#         self.output_1 = nn.Linear(d_model, d_model // 2)
#         self.output_2 = nn.Linear(d_model // 2, 2)
    
#     def forward(self, x, y, z):
#         x = F.relu(self.conv1(x))
#         x = self.conv1_bn(self.pool1(x))
#         x = self.conv2_bn(F.relu(self.conv2(x)))
#         x = self.conv3(x)
#         x = x.reshape(x.shape[0], -1)
#         y = y.reshape(y.shape[0], -1)
#         z = z.reshape(z.shape[0], -1)
#         # print(x.shape,y.shape,z.shape)
#         x = torch.cat([x,y,z],dim=1)
#         output = F.relu(self.input_linear(x))
#         output = F.relu(self.output_1(output))
#         output = self.output_2(output)
#         return output
