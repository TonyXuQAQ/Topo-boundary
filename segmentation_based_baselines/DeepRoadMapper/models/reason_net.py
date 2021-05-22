import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

class ConvReLU(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_sz, stride=1, relu=True, do=False, bn=True):
            super(ConvReLU, self).__init__()
            padding = int((kernel_sz - 1) / 2)  # same spatial size by default
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_sz, stride, padding=padding)
            self.relu = nn.ReLU(inplace=True) if relu else None
            self.bn = nn.BatchNorm2d(out_ch) if bn else None
            self.do = nn.Dropout(p=0.5) if do else None

        def forward(self, x):
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)
            if self.do is not None:
                x = self.do(x)
            return x

class ReasonNet(nn.Module):
    def __init__(self,input_channels=6):
        super().__init__()
        
        self.layer1 = ConvReLU(input_channels,32,3,stride=2,bn=False)
        self.layer2 = ConvReLU(32,64,3,stride=2)
        self.layer3 = ConvReLU(64,128,3,stride=2,do=True)
        self.layer4 = ConvReLU(128,128,3,stride=2,do=True)
        self.layer5 = ConvReLU(128,128,3,stride=2,do=True)
        self.layer6 = ConvReLU(128,128,3,stride=2,do=True)
        self.layer7 = ConvReLU(128,128,3,stride=2,do=True)
        self.layer8 = ConvReLU(128,128,3,stride=2)
        self.output_layer = ConvReLU(128,2,3,stride=2)


    def forward(self,x):
        # print(x.shape)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        output = self.output_layer(x8)
        output = output.reshape(output.shape[0],-1)

        return output


if __name__ == '__main__':
    tensor = torch.rand(1,6,320,320)
    net = ReasonNet()
    output = net(tensor)
