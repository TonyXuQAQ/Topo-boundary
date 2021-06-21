import torch
import torch.nn as nn
import torch.nn.functional as F
from .spatial_argmax import *
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

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

class IRC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.irc = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels, out_channels  , kernel_size=3, padding=1, bias=False)
        )
        
    def forward(self, x):
        return self.irc(x)

class IRUC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ir = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )     
        self.conv = nn.Conv2d(in_channels, out_channels  , kernel_size=3, padding=1, bias=False)
        
    def forward(self, x , h, w):
        upsample =  F.interpolate(self.ir(x), size=(h,w), mode='bilinear', align_corners=True)  
        return self.conv(upsample)


class RNNCell(nn.Module):
    def __init__(self, num_chars, device,num_hidden=32):
        super().__init__()
        self.num_chars = num_chars
        self.num_hidden = num_hidden
        self.device = device
        self.conv = nn.Conv2d(in_channels=self.num_chars + self.num_hidden,
                              out_channels=2 * self.num_hidden,
                              kernel_size=3,
                              padding=1)
        self.conv_output = nn.Conv2d(in_channels=self.num_hidden,
                              out_channels= self.num_chars,
                              kernel_size=3,
                              padding=1)
        self.relu = nn.ReLU()
        self._init_weights()
        
    def _init_weights(self):
        for param in self.parameters():
            param.requires_grad_(True)
            
            if param.data.ndimension() >= 2:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)
                
    def forward(self, input, h):
        tensor = torch.cat([input,h],dim=1)
        combined_conv = self.conv(tensor)
        c_o, c_h = torch.split(combined_conv, self.num_hidden, dim=1)
        hidden_state = h + self.relu(c_o+c_h)
        output = self.conv_output(hidden_state)
        return output, hidden_state

class ConvRNNNet(nn.Module):
    def __init__(self,tensor_w,tensor_c,device):
        super().__init__()
        self.w = tensor_w
        self.device = device
        self.rnn_cell = RNNCell(tensor_c,device)

    def forward(self,x):
        hidden_state = torch.zeros(1,32,x[0].shape[2],x[0].shape[3]).to(self.device)
        for i in range(3):
            input = x[i]
            output, hidden_state = self.rnn_cell(input.to(self.device),hidden_state.to(self.device))
        return output


class DAGMapperEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.input_layer = nn.Conv2d(4,16,3,stride=2)
        self.max_pool = nn.MaxPool2d(2, stride=2,padding=1)

        self.res_layer1 = BasicBlock(16,16,stride=2)
        self.res_layer2 = BasicBlock(16,16,stride=2)

        self.res_layer3 = BasicBlock(16,16,stride=1)
        self.res_layer4 = BasicBlock(16,16,stride=1)

        self.res_layer5 = BasicBlock(16,32,stride=2)
        self.res_layer6 = BasicBlock(32,32,stride=1)

        self.res_layer7 = BasicBlock(32,64,stride=2)
        self.res_layer8 = BasicBlock(64,64,stride=1)

        self.res_layer9 = BasicBlock(64,128,stride=2)
        self.res_layer10 = BasicBlock(128,128,stride=1)

        self.IRC1 = IRC(128,64)
        self.IRUC1 = IRUC(64,64)
        self.IRC2 = IRC(64,64)

        self.IRC3 = IRC(128,32)
        self.IRUC2 = IRUC(32,32)
        self.IRC4 = IRC(32,32)

        self.IRC5 = IRC(64,16)
        self.IRUC3 = IRUC(16,16)
        self.IRC6 = IRC(16,16)

        self.IRC7 = IRC(32,16)
        self.IRUC4 = IRUC(16,16)
        self.IRC8 = IRC(16,16)

        self.IRUC5 = IRUC(16,16)
        self.final_layer = nn.Conv2d(16,8,1,1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _upsample(self,x,size):
        return F.interpolate(x,size=(size,size),mode='bilinear',align_corners=True)

    def forward(self,x):
        x1 = self.max_pool(self.input_layer(x))
        x2 = self.res_layer2(self.res_layer1(x1))
        x3 = self.res_layer4(self.res_layer3(x2))
        x4 = self.res_layer6(self.res_layer5(x3))
        x5 = self.res_layer8(self.res_layer7(x4))
        x6 = self.res_layer10(self.res_layer9(x5))

        x7 = self.IRC2(self.IRUC1(self.IRC1(x6),x5.shape[2],x5.shape[3])) 
        x7 = torch.cat([x5,x7],dim=1)

        x8 = self.IRC4(self.IRUC2(self.IRC3(x7),x4.shape[2],x4.shape[3])) 
        x8 = torch.cat([x4,x8],dim=1)

        x9 = self.IRC6(self.IRUC3(self.IRC5(x8),x3.shape[2],x3.shape[3]))
        x9 = torch.cat([x3,x9],dim=1)

        x10 = self.IRC8(self.IRUC4(self.IRC7(x9),x.shape[2]//8,x.shape[3]//8))
        x11 = self.final_layer(self.IRUC5(x10,x.shape[2]//4,x.shape[3]//4))

        return x11



class DAGMapperDTH(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8,16,1,1)
        self.res_layer1 = BasicBlock(16,16,stride=1)
        self.res_layer2 = BasicBlock(16,16,stride=1)
        self.res_layer3 = BasicBlock(16,16,stride=1)
        self.res_layer4 = BasicBlock(16,16,stride=1)
        self.conv2 = nn.Conv2d(16,1,1,1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.res_layer1(x1)
        x3 = self.res_layer2(x2)
        x4 = self.res_layer3(x3)
        x5 = self.res_layer4(x4)
        x6 = self.conv2(x5)

        return x6


class DAGMapperSH(nn.Module):
    def __init__(self,env):
        super().__init__()
        self.convRNN = ConvRNNNet(env.crop_size,10,env.args.device)
        self.conv = nn.Conv2d(10,16,1,1)
        self.res_layer1 = BasicBlock(16,16,stride=1)
        self.res_layer2 = BasicBlock(16,16,stride=2)
        self.res_layer3 = BasicBlock(16,32,stride=2)
        self.res_layer4 = BasicBlock(32,32,stride=2)
        self.max_pool = nn.MaxPool2d(2,stride=2)
        self.linear1 = nn.Linear(2048,50)
        self.linear2 = nn.Linear(50,2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        # x0 = self.convRNN(x)
        x1 = self.conv(x)
        x2 = self.res_layer4(self.res_layer3(self.res_layer2(self.res_layer1(x1))))
        x3 = self.max_pool(x2)
        x3 = x3.reshape(x3.shape[0],-1)
        x4 = nn.ReLU(inplace=True)(self.linear1(x3))
        x5 = self.linear2(x4)
        return x5

class DAGMapperDH(nn.Module):
    def __init__(self,env):
        super().__init__()
        self.convRNN = ConvRNNNet(env.crop_size,10,env.args.device)
        self.conv = nn.Conv2d(10,16,1,1)
        self.res_layer1 = BasicBlock(16,16,stride=1)
        self.res_layer2 = BasicBlock(16,16,stride=1)
        self.res_layer3 = BasicBlock(16,32,stride=2)
        self.res_layer4 = BasicBlock(32,32,stride=1)
        self.max_pool = nn.MaxPool2d(2,stride=2)
        self.linear1 = nn.Linear(32*16*16*4,50)
        self.linear2 = nn.Linear(50,2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        # x0 = self.convRNN(x)
        x1 = self.conv(x)
        x2 = self.res_layer4(self.res_layer3(self.res_layer2(self.res_layer1(x1))))
        x3 = self.max_pool(x2)
        x3 = x3.reshape(x3.shape[0],-1)
        x4 = nn.ReLU(inplace=True)(self.linear1(x3))
        x5 = self.linear2(x4)
        x6 = x5.norm(p=2, dim=1, keepdim=True)
        x7 = x5.div(x6.expand_as(x5))
        return x7

class DAGMapperPH(nn.Module):
    def __init__(self, device,block=Bottleneck, num_blocks=[2,4,23,3],n_channels=10,n_classes=1,params=[1,64,24]):
        super(DAGMapperPH, self).__init__()
        self.in_planes = 64
        self.device=device
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.params = params

        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 8, kernel_size=1, stride=1, padding=0)
        self.output = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.gn1 = nn.GroupNorm(128, 128) 
        self.gn2 = nn.GroupNorm(256, 256)

        # self.output = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        self.final = nn.Conv2d(8, self.params[0], kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        # c5 = self.layer4(c4)
        # print(c1.shape,c2.shape,c3.shape,c4.shape,c5.shape)
        # Top-down
        p4 = self.toplayer(c4)
        # p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # print(p2.shape,p3.shape,p4.shape,p5.shape)
        # Semantic
        _, _, h, w = p2.size()
        # 256->256
        # s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        # # 256->256
        # s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)
        # # 256->128
        # s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        # 256->256
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w)
        # 256->128
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))

        s = self.conv3(s2 + s3 + s4 )
        
        prob_map = self._upsample(self.final(s), x.shape[-2], x.shape[-1])

        # next_vertex = soft_argmax(prob_map,self.device)
        return prob_map
# class DAGMapperPH(nn.Module):
#     def __init__(self,env):
#         super().__init__()
#         self.convRNN = ConvRNNNet(env.crop_size,10,env.args.device)
#         self.input_layer = nn.Conv2d(10,16,3,stride=2)
#         self.max_pool = nn.MaxPool2d(2, stride=2,padding=1)
#         self.device = env.args.device

#         self.res_layer1 = BasicBlock(16,16,stride=2)
#         self.res_layer2 = BasicBlock(16,16,stride=2)

#         self.res_layer3 = BasicBlock(16,16,stride=1)
#         self.res_layer4 = BasicBlock(16,16,stride=1)

#         self.res_layer5 = BasicBlock(16,32,stride=2)
#         self.res_layer6 = BasicBlock(32,32,stride=1)

#         self.res_layer7 = BasicBlock(32,64,stride=1)
#         self.res_layer8 = BasicBlock(64,64,stride=1)

#         self.res_layer9 = BasicBlock(64,128,stride=1)
#         self.res_layer10 = BasicBlock(128,128,stride=1)

#         self.IRC1 = IRC(128,64)
#         self.IRUC1 = IRUC(64,64)
#         self.IRC2 = IRC(64,64)

#         self.IRC3 = IRC(128,32)
#         self.IRUC2 = IRUC(32,32)
#         self.IRC4 = IRC(32,32)

#         self.IRC5 = IRC(64,16)
#         self.IRUC3 = IRUC(16,16)
#         self.IRC6 = IRC(16,16)

#         self.IRC7 = IRC(32,16)
#         self.IRUC4 = IRUC(16,16)
#         self.IRC8 = IRC(16,16)

#         self.IRUC5 = IRUC(16,16)
#         self.res_layer11 = BasicBlock(16,1,stride=1)
#         self.final_layer = nn.Conv2d(1,1,1,1)
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _upsample(self,x,size):
#         return F.interpolate(x,size=(size,size),mode='bilinear',align_corners=True)


#     def forward(self,x):
#         # x0 = self.convRNN(x)
#         x1 = self.max_pool(self.input_layer(x))
#         x2 = self.res_layer2(self.res_layer1(x1))
#         x3 = self.res_layer4(self.res_layer3(x2))
#         x4 = self.res_layer6(self.res_layer5(x3))
#         x5 = self.res_layer8(self.res_layer7(x4))
#         x6 = self.res_layer10(self.res_layer9(x5))

#         x7 = self.IRC2(self.IRUC1(self.IRC1(x6),x5.shape[2],x5.shape[3])) 
#         x7 = torch.cat([x5,x7],dim=1)

#         x8 = self.IRC4(self.IRUC2(self.IRC3(x7),x4.shape[2],x4.shape[3])) 
#         x8 = torch.cat([x4,x8],dim=1)

#         x9 = self.IRC6(self.IRUC3(self.IRC5(x8),x3.shape[2],x3.shape[3]))
#         x9 = torch.cat([x3,x9],dim=1)
#         x10 = self.IRC8(self.IRUC4(self.IRC7(x9),x.shape[2]//2,x.shape[3]//2))
#         x11 = self.final_layer(self.res_layer11(self.IRUC5(x10,x.shape[2],x.shape[3])))
#         # x12 = soft_argmax(x11,self.device)
#         return x11