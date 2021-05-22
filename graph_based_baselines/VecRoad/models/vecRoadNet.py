from pathlib import Path

import torch
import torch.nn.init
import torch.utils.model_zoo as model_zoo
from torch import nn
import torch.nn.functional as F
from .res2net import res2net50_v1b_26w_4s

upsample = lambda x, scale: \
    F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)

upsample_size = lambda x, size: \
    F.interpolate(x, size=(size,size), mode='bilinear', align_corners=True)

class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sz, stride=1, relu=True, pd=True, bn=False):
        super(ConvReLU, self).__init__()
        padding = int((kernel_sz - 1) / 2) if pd else 0  # same spatial size by default
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_sz, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch) if bn else None  # eps=0.001, momentum=0, affine=True
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, ft_in, ft_out, out_ch):
        super(DecoderBlock, self).__init__()

        self.ft_conv = nn.Sequential(
            ConvReLU(ft_in, ft_out, 1))
        self.conv = nn.Sequential(
            ConvReLU(out_ch + ft_out, (out_ch + ft_out) // 2, 3),
            nn.Conv2d((out_ch + ft_out) // 2, out_ch, 3, padding=1))

    def forward(self, ft_cur, ft_pre):
        ft_cur = self.ft_conv(ft_cur)
        x = torch.cat((ft_cur, ft_pre), dim=1)
        x = self.conv(x)
        return x


class Hourglass(nn.Module):

    def __init__(self, input_ch, output_ch, ch=[32, 32, 32, 32]):
        super(Hourglass, self).__init__()
        ch = [input_ch] + ch
        self.encoder_1 = nn.Sequential(
            ConvReLU(ch[0], (ch[0]+ch[1])//2, 3),
            ConvReLU((ch[0]+ch[1])//2, ch[1], 3)
        )
        self.encoder_2 = nn.Sequential(
            ConvReLU(ch[1], (ch[1]+ch[2])//2, 3),
            ConvReLU((ch[1]+ch[2])//2, ch[2], 3)
        )
        self.encoder_3 = nn.Sequential(
            ConvReLU(ch[2], (ch[2]+ch[3])//2, 3),
            ConvReLU((ch[2]+ch[3])//2, ch[3], 3)
        )
        self.encoder_4 = nn.Sequential(
            ConvReLU(ch[3], (ch[3]+ch[4])//2, 3),
            ConvReLU((ch[3]+ch[4])//2, ch[4], 3)
        )
        self.encoder_5 = nn.Sequential(
            ConvReLU(ch[4], ch[4], 3),
            ConvReLU(ch[4], ch[4], 3)
        )
        self.decoder_4 = nn.Sequential(
            ConvReLU(ch[4], (ch[3]+ch[4])//2, 3),
            ConvReLU((ch[3]+ch[4])//2, ch[3], 3)
        )
        self.decoder_3 = nn.Sequential(
            ConvReLU(ch[3], (ch[2]+ch[3])//2, 3),
            ConvReLU((ch[2]+ch[3])//2, ch[2], 3)
        )
        self.decoder_2 = nn.Sequential(
            ConvReLU(ch[2], (ch[1]+ch[2])//2, 3),
            ConvReLU((ch[1]+ch[2])//2, ch[1], 3)
        )
        self.decoder_1 = nn.Sequential(
            ConvReLU(ch[1], (ch[0]+ch[1])//2, 3),
            ConvReLU((ch[0]+ch[1])//2, output_ch, 3)
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_1 = self.encoder_1(x)
        encoder_1_pool = self.maxpool(encoder_1)
        encoder_2 = self.encoder_2(encoder_1_pool)
        encoder_2_pool = self.maxpool(encoder_2)
        encoder_3 = self.encoder_3(encoder_2_pool)
        encoder_3_pool = self.maxpool(encoder_3)
        encoder_4 = self.encoder_4(encoder_3_pool)
        encoder_4_pool = self.maxpool(encoder_4)
        encoder_5 = self.encoder_5(encoder_4_pool)
        decoder_5_up = upsample(encoder_5, 2) + encoder_4
        decoder_4 = self.decoder_4(decoder_5_up)
        decoder_4_up = upsample(decoder_4, 2) + encoder_3
        decoder_3 = self.decoder_3(decoder_4_up)
        decoder_3_up = upsample(decoder_3, 2) + encoder_2
        decoder_2 = self.decoder_2(decoder_3_up)
        decoder_2_up = upsample(decoder_2, 2) + encoder_1
        decoder_1 = self.decoder_1(decoder_2_up)
        return decoder_1


class RPNet(nn.Module):

    def __init__(self):
        super(RPNet, self).__init__()


        self.conv_2_side = ConvReLU(256, 128, 3, 1, bn=True)
        self.conv_3_side = ConvReLU(512, 128, 3, 1, bn=True)
        self.conv_4_side = ConvReLU(1024, 128, 3, 1, bn=True)
        self.conv_5_side = ConvReLU(2048, 128, 3, 1, bn=True)
        self.conv_fuse = ConvReLU(512, 128, 3, 1, bn=True)
        self.avgpool4 = nn.AvgPool2d(4, 4).cuda()
        
        self.road_seg = nn.Sequential(
            ConvReLU(128, 64, 3, 1, bn=True),
            ConvReLU(64, 64, 1, 1, bn=True)
        )
        self.conv_road_final = nn.Conv2d(64, 1, 1, 1, 0)

        self.junc_seg = nn.Sequential(
            ConvReLU(128, 64, 3, 1, bn=True),
            ConvReLU(64, 64, 1, 1, bn=True)
        )
        self.conv_junc_final = nn.Conv2d(64, 1, 1, 1, 0)
        self.fuse_module = Hourglass(
            128 + 64 ,  # 353
            32, [128, 128, 128, 128])

        self.ft_chs = [1024, 512, 256, 64]
        self.decoders = nn.ModuleList([
            DecoderBlock(self.ft_chs[0], 32, 32),
            DecoderBlock(self.ft_chs[1], 32, 32),
            DecoderBlock(self.ft_chs[2], 32, 32),
            DecoderBlock(self.ft_chs[3], 32, 32),
        ])
        self.next_step_final = nn.Conv2d(32, 1, 1, 1, 0)
        self.conv_final = nn.Conv2d(32, 1, 3, 1, 1)

        self.init_weights()
        ## first init_weights for added parts, then init res2net
        res2net = res2net50_v1b_26w_4s(pretrained=False)
        self.stage_1 = nn.Sequential(
            res2net.conv1,
            res2net.relu)
        self.maxpool = res2net.maxpool
        self.stage_2 = res2net.layer1
        self.stage_3 = res2net.layer2
        self.stage_4 = res2net.layer3
        self.stage_5 = res2net.layer4

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, aerial_image, NUM_TARGETS=None, test=False):
        stage_1 = self.stage_1(aerial_image)     #1/2
        stage_1_down = self.maxpool(stage_1)     #1/4

        stage_2 = self.stage_2(stage_1_down)     #1/4
        stage_2_side = self.conv_2_side(stage_2) #1/4

        stage_3 = self.stage_3(stage_2)          #1/8
        stage_3_side = self.conv_3_side(stage_3) #1/8
        stage_3_side = upsample(stage_3_side, 2) #1/4

        stage_4 = self.stage_4(stage_3)          #1/8
        stage_4_side = self.conv_4_side(stage_4) #1/8
        stage_4_side = upsample(stage_4_side, 2) #1/4

        stage_5 = self.stage_5(stage_4)          #1/8
        stage_5_side = self.conv_5_side(stage_5) #1/8
        stage_5_side = upsample(stage_5_side, 2) #1/4

        stage_fuse = [stage_2_side, stage_3_side, stage_4_side, stage_5_side]
        stage_fuse = torch.cat(stage_fuse, dim=1)
        stage_fuse = self.conv_fuse(stage_fuse)

        road_fts = self.road_seg(stage_fuse)
        road_final = upsample(self.conv_road_final(road_fts),4)

        stage_fuse = torch.cat(
            [stage_fuse, road_fts], dim=1)
        
        next_step = self.fuse_module(stage_fuse)

        decoded_ft_4 = self.decoders[0](upsample(stage_4, 2), next_step)
        decoded_ft_3 = self.decoders[1](upsample(stage_3, 2), decoded_ft_4)
        decoded_ft_2 = self.decoders[2](upsample(stage_2, 2), upsample(decoded_ft_3, 2))
        decoded_ft_1 = self.decoders[3](upsample(stage_1, 2), upsample(decoded_ft_2, 2))

        decoded_ft_1 = self.conv_final(decoded_ft_1)
        next_point = decoded_ft_1
        return  road_final, next_point


def VecRoadNet():
    return RPNet()


