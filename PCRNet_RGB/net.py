# Copyright 2023 Chen Zhou
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/python3
# coding=utf-8
#!/usr/bin/python3
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.AvgPool2d):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out + x, inplace=True)


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('/home/work3/PCRNet/PCRNet_RGB/script/res/resnet50-19c8e357.pth'), strict=False)


# class SEF(nn.Module):
#     def __init__(self, inc=128, outc=64):
#         super(SEF, self).__init__()
#         self.conv0 = nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=1, dilation=1)
#         self.conv1 = nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=3, dilation=3)
#         self.conv2 = nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=5, dilation=5)
#         self.conv3 = nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=7, dilation=7)
#         self.outconv = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x,f1):
#         resl = x + self.conv0(x*f1)
#         y = self.conv1(x*f1)
#         resl = resl + y
#         y = self.conv2(x*f1)
#         resl = resl + y
#         y = self.conv3(x*f1)
#         resl = resl + y
#         resl = F.relu(resl, inplace=True)
#         resl = self.outconv(resl)
#         return resl
#
#     def initialize(self):
#         weight_init(self)
class SEF(nn.Module):
    def __init__(self, inc=128, outc=64):
        super(SEF, self).__init__()
        self.conv0 = nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1 = nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv2 = nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=5, dilation=5)
        self.conv3 = nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=7, dilation=7)
        self.outconv = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.conv0(x)
        resl = x + y
        y = self.conv1(x+y)
        resl = resl + y
        y = self.conv2(x+y)
        resl = resl + y
        y = self.conv3(x+y)
        resl = resl + y
        resl = F.relu(resl, inplace=True)
        resl = self.outconv(resl)
        return resl

    def initialize(self):
        weight_init(self)
class CMF(nn.Module):  # cross modal fuse module
    def __init__(self, inc=128, outc=64):
        super(CMF, self).__init__()
        self.SEF = SEF(inc, outc)
        # self.RGB = nn.Sequential(nn.Conv2d(64, outc, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
        #                          nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.Depth = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(outc),
                                   nn.ReLU(inplace=True), nn.Conv2d(outc, 1, kernel_size=3, padding=1))
        self.outconv = nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1)

    def forward(self, f):
        #d1 = self.RGB(r)
        s1 = torch.sigmoid(self.Depth(f))
        f1 = self.SEF(f,s1)
        # s1*f1代表两者都有的特征，也就是绝对正确的特征，需要进行强调，而d1代表着细节特征，是需要补充上去的。
        # 最后+上f1是为了不让显著性特征丢失，
        # 这一步的目的是强调绝对正确的特征，补充细节特征，以及防止融合特征丢失。
        #f_out = torch.mul(torch.mul(f1, torch.sigmoid(s1)), torch.sigmoid(d1)) + f1

        resl = self.outconv(f1)
        return resl

    def initialize(self):
        weight_init(self)
class CMFE(nn.Module):  # cross modal fuse module#This module is valid
    def __init__(self, inc=128, outc=64):
        super(CMFE, self).__init__()
        self.SEF = SEF(inc, 64)
        self.outconv = nn.Conv2d(64, outc, kernel_size=3, stride=1, padding=1)
        self.fuse = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))
    def forward(self, f):

        f2 = torch.sigmoid(self.fuse(f))
        f1 = self.SEF(f)
        f1 = f1 + f1 * f2
        resl = self.outconv(f1)
        return resl

    def initialize(self):
        weight_init(self)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.rgbconv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgbbn0 = nn.BatchNorm2d(64)
        self.rgbconv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgbbn1 = nn.BatchNorm2d(64)
        self.rgbconv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgbbn2 = nn.BatchNorm2d(64)
        self.rgbconv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgbbn3 = nn.BatchNorm2d(64)
        self.rgbconv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgbbn4 = nn.BatchNorm2d(64)

        self.depthconv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.depthbn0 = nn.BatchNorm2d(64)
        self.depthconv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.depthbn1 = nn.BatchNorm2d(64)
        self.depthconv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.depthbn2 = nn.BatchNorm2d(64)
        self.depthconv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.depthbn3 = nn.BatchNorm2d(64)
        self.depthconv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.depthbn4 = nn.BatchNorm2d(64)

        self.MFF0 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.MFF1 = SEF(192, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.MFF2 = SEF(192, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.MFF3 = SEF(192, 64)
        self.bn3 = nn.BatchNorm2d(64)
        # self.MFF4 = CMF(192, 64)
        # self.bn4 = nn.BatchNorm2d(64)

    def forward(self, input1, input2):  # input1:detail  input2:body
        rgb0 = F.relu(self.rgbbn0(self.rgbconv0(input1[0])), inplace=True)
        depth0 = F.relu(self.depthbn0(self.depthconv0(input2[0])), inplace=True)
        out0 = F.relu(self.bn0(self.MFF0(torch.cat([rgb0, depth0], dim=1))), inplace=True)

        rgb0_up = F.interpolate(rgb0, size=input1[1].size()[2:], mode='bilinear')
        depth0_up = F.interpolate(depth0, size=input1[1].size()[2:], mode='bilinear')
        out0_up = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        rgb1 = F.relu(self.rgbbn1(self.rgbconv1(rgb0_up + input1[1] + out0_up)), inplace=True)
        depth1 = F.relu(self.depthbn1(self.depthconv1(depth0_up + input2[1] + out0_up)), inplace=True)
        out1 = F.relu(self.bn1(self.MFF1(torch.cat([rgb1, depth1,out0_up], dim=1))), inplace=True)

        rgb1_up = F.interpolate(rgb1, size=input1[2].size()[2:], mode='bilinear')
        depth1_up = F.interpolate(depth1, size=input1[2].size()[2:], mode='bilinear')
        out1_up = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        rgb2 = F.relu(self.rgbbn2(self.rgbconv2(rgb1_up + input1[2] + out1_up)), inplace=True)
        depth2 = F.relu(self.depthbn2(self.depthconv2(depth1_up + input2[2] + out1_up)), inplace=True)
        out2 = F.relu(self.bn2(self.MFF2( torch.cat([rgb2, depth2,out1_up], dim=1))), inplace=True)

        rgb2_up = F.interpolate(rgb2, size=input1[3].size()[2:], mode='bilinear')
        depth2_up = F.interpolate(depth2, size=input1[3].size()[2:], mode='bilinear')
        out2_up = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        rgb3 = F.relu(self.rgbbn3(self.rgbconv3(rgb2_up + input1[3] + out2_up)), inplace=True)
        depth3 = F.relu(self.depthbn3(self.depthconv3(depth2_up + input2[3] + out2_up)), inplace=True)
        out3 = F.relu(self.bn3(self.MFF3(torch.cat([rgb3, depth3,out2_up], dim=1))), inplace=True)

        # rgb3_up = F.interpolate(rgb3, size=input1[4].size()[2:], mode='bilinear')
        # depth3_up = F.interpolate(depth3, size=input1[4].size()[2:], mode='bilinear')
        # out3_up = F.interpolate(out3, size=input1[4].size()[2:], mode='bilinear')
        # rgb4 = F.relu(self.rgbbn4(self.rgbconv4(rgb3_up + input1[4] + out3_up)), inplace=True)
        # depth4 = F.relu(self.depthbn4(self.depthconv4(depth3_up+input2[4] + out3_up)), inplace=True)
        # out4 = F.relu(self.bn4(self.MFF4(torch.cat([rgb4, depth4], dim=1))), inplace=True)

        return rgb3, depth3, out3

    def initialize(self):
        weight_init(self)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.conv1 = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        self.CMF1=nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.CMF2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.CMF3=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.CMF4=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)#CMF(64,64)
        self.bn4 = nn.BatchNorm2d(64)
        # self.CMF5=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn5 = nn.BatchNorm2d(64)

        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(64)
        self.conv4b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(64)

        self.conv1d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1d = nn.BatchNorm2d(64)
        self.conv2d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(64)
        self.conv3d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3d = nn.BatchNorm2d(64)
        self.conv4d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4d = nn.BatchNorm2d(64)

    def forward(self, out1):
        out1 = F.relu(self.bn1(self.CMF1(out1)), inplace=True)
        out2 = F.max_pool2d(out1, kernel_size=2, stride=2)
        out2 = F.relu(self.bn2(self.CMF2(out2)), inplace=True)
        out3 = F.max_pool2d(out2, kernel_size=2, stride=2)
        out3 = F.relu(self.bn3(self.CMF3(out3)), inplace=True)
        out4 = F.max_pool2d(out3, kernel_size=2, stride=2)
        out4 = F.relu(self.bn4(self.CMF4(out4)), inplace=True)

        out1b = F.relu(self.bn1b(self.conv1b(out1)), inplace=True)
        out2b = F.relu(self.bn2b(self.conv2b(out2)), inplace=True)
        out3b = F.relu(self.bn3b(self.conv3b(out3)), inplace=True)
        out4b = F.relu(self.bn4b(self.conv4b(out4)), inplace=True)

        out1d = F.relu(self.bn1d(self.conv1d(out1)), inplace=True)
        out2d = F.relu(self.bn2d(self.conv2d(out2)), inplace=True)
        out3d = F.relu(self.bn3d(self.conv3d(out3)), inplace=True)
        out4d = F.relu(self.bn4d(self.conv4d(out4)), inplace=True)
        return out4b, out3b, out2b, out1b, out4d, out3d, out2d, out1d

    def initialize(self):
        weight_init(self)


class LDF(nn.Module):
    def __init__(self, cfg):
        super(LDF, self).__init__()
        self.cfg = cfg
        self.bkbone = ResNet(cfg)
        self.conv5b = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv5d = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4d = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3d = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2d = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.decoderd = Decoder()
        self.linearb = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.lineard = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.linear = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))

        self.initialize()

    def forward(self, x, shape=None):
        out1, out2, out3, out4, out5 = self.bkbone(x)
        out2b, out3b, out4b, out5b = self.conv2b(out2), self.conv3b(out3), self.conv4b(out4), self.conv5b(out5)
        out2d, out3d, out4d, out5d = self.conv2d(out2), self.conv3d(out3), self.conv4d(out4), self.conv5d(out5)

        d1, b1, out1 = self.decoder([out5b, out4b, out3b, out2b], [out5d, out4d, out3d, out2d])

        # outb5, outb4, outb3, outb2, outd5, outd4, outd3, outd2 = self.encoder(torch.cat([d1, b1, out1], dim=1))
        # outb5, outb4, outb3, outb2, outd5, outd4, outd3, outd2 = outb5 + out5b, outb4 + out4b, outb3 + out3b, outb2 + out2b, outd5 + out5d, outd4 + out4d, outd3 + out3d, outd2 + out2d
        # d2, b2, out2 = self.decoder([outb5, outb4, outb3, outb2], [outd5, outd4, outd3, outd2])

        # outb5, outb4, outb3, outb2, outd5, outd4, outd3, outd2 = self.encoder(torch.cat([d2, b2, out2], dim=1))
        # outb5, outb4, outb3, outb2, outd5, outd4, outd3, outd2 = outb5 + out5b, outb4 + out4b, outb3 + out3b, outb2 + out2b, outd5 + out5d, outd4 + out4d, outd3 + out3d, outd2 + out2d
        # d3, b3, out3 = self.decoder([outb5, outb4, outb3, outb2], [outd5, outd4, outd3, outd2])
        # f1 = torch.cat([d2, b2, out2 ], dim=1)
        # outb5, outb4, outb3, outb2, outd5, outd4, outd3, outd2 = self.encoder(torch.cat([d3, b3, out3], dim=1))
        # outb5, outb4, outb3, outb2, outd5, outd4, outd3, outd2 = outb5 + out5b, outb4 + out4b, outb3 + out3b, outb2 + out2b, outd5 + out5d, outd4 + out4d, outd3 + out3d, outd2 + out2d
        # d4, b4, out4 = self.decoder([outb5, outb4, outb3, outb2], [outd5, outd4, outd3, outd2])

        if shape is None:
            shape = x.size()[2:]
        out1 = F.interpolate(self.linear(out1), size=shape, mode='bilinear')
        b1 = F.interpolate(self.linearb(b1), size=shape, mode='bilinear')
        d1 = F.interpolate(self.lineard(d1), size=shape, mode='bilinear')

        # out2 = F.interpolate(self.linear(out2), size=shape, mode='bilinear')
        # b2 = F.interpolate(self.linearb(b2), size=shape, mode='bilinear')
        # d2 = F.interpolate(self.lineard(d2), size=shape, mode='bilinear')

        #out3 = F.interpolate(self.linear(out3), size=shape, mode='bilinear')
        # b3 = F.interpolate(self.linearb(b3), size=shape, mode='bilinear')
        # d3 = F.interpolate(self.lineard(d3), size=shape, mode='bilinear')

        # out4 = F.interpolate(self.linear(out4), size=shape, mode='bilinear')
        # b4 = F.interpolate(self.linearb(b4), size=shape, mode='bilinear')
        # d4 = F.interpolate(self.lineard(d4), size=shape, mode='bilinear')

        return  d1, b1, out1

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
