import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

class ConvReLU(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_sz, stride=1, relu=True, pd=True, bn=False):
            super(ConvReLU, self).__init__()
            padding = int((kernel_sz - 1) / 2) if pd else 0  # same spatial size by default
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_sz, stride, padding=padding)
            self.relu = nn.ReLU(inplace=True) if relu else None

        def forward(self, x):
            x = self.conv(x)
            if self.relu is not None:
                x = self.relu(x)
            return x

class RoadTracerNet(nn.Module):
    def __init__(self,input_channels=5):
        super().__init__()
        
        self.layer1 = ConvReLU(input_channels,128,3,stride=2)
        self.layer2 = ConvReLU(128,128,3,stride=1)
        self.layer3 = ConvReLU(128,256,3,stride=2)
        self.layer4 = ConvReLU(256,256,3,stride=1)
        self.layer5 = ConvReLU(256,256,3,stride=1)
        self.layer6 = ConvReLU(256,256,3,stride=1)
        self.layer7 = ConvReLU(256,512,3,stride=2)
        self.layer8 = ConvReLU(512,512,3,stride=1)
        self.layer9 = ConvReLU(512,512,3,stride=2)
        self.layer10 = ConvReLU(512,512,3,stride=1)
        self.layer11 = ConvReLU(512,512,3,stride=2)
        self.layer12 = ConvReLU(512,512,3,stride=1)
        self.layer13 = ConvReLU(512,512,3,stride=1)
        self.layer14 = ConvReLU(512,512,3,stride=2)
        self.layer15 = ConvReLU(512,512,3,stride=1)
        self.layer16 = ConvReLU(512,512,3,stride=1)
        self.layer17 = ConvReLU(512,512,3,stride=2)

        self.softmax = nn.Softmax(dim=1)

        self.detect_outputs_layer = ConvReLU(256,1,3,stride=1,relu=False)
        self.action_outputs_layer = ConvReLU(512,2,3,stride=2,relu=False)
        self.angle_outputs_layer = ConvReLU(512,64,3,stride=2,relu=False)


    def forward(self,x,y):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)
        x10 = self.layer10(x9)
        x11 = self.layer11(x10)
        x12 = self.layer12(x11)
        x13 = self.layer13(x12)
        x14 = self.layer14(x13)
        x15 = self.layer15(x14)
        x16 = self.layer16(x15)
        x17 = self.layer17(x16)

        detection_pre_outputs = self.detect_outputs_layer(x6)
        detection_outputs = F.interpolate(detection_pre_outputs, size=(x.shape[2],x.shape[3]), mode='bilinear', align_corners=True)#self.softmax(detection_pre_outputs)

        action_outputs = self.action_outputs_layer(x17)
        angle_outputs = self.angle_outputs_layer(x17)

        return detection_outputs, angle_outputs, action_outputs


if __name__ == '__main__':
    tensor = torch.rand(1,4,127,127)
    net = RoadTracerNet()
    detection, action, angle = net(tensor)
    print(detection.shape,action.shape,angle.shape)
