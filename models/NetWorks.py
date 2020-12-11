"""
some networks
edit by hichens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torchvision.models import vgg16


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        
        ## More flexiable structrue
        net = []
        channels = [32, 64, 128]
        for i in range(3):
            if i == 0:
                net.append(ConvLayer(in_channels=3, out_channels=channels[i], kernel_size=9, stride=1))
            else:
                net.append(ConvLayer(in_channels=channels[i-1], out_channels=channels[i], kernel_size=3, stride=2))

            net.append(nn.InstanceNorm2d(channels[i]))
            net.append(nn.ReLU())
        
        for i in range(5):
            net.append(ResidualBlock(128))
        
        channels = [128, 64, 32]
        for i in range(2):
            net.append(UpsampleConvLayer(in_channels=channels[i], out_channels=channels[i+1], kernel_size=3, stride=1, upsample=2))
            net.append(nn.InstanceNorm2d(channels[i+1]))
            net.append(nn.ReLU())
        net.append(ConvLayer(in_channels=32, out_channels=3, kernel_size=9, stride=1))
        self.net = nn.Sequential(*net)

        ## Normal structrue
        # self.net = nn.Sequential(
        #     ConvLayer(in_channels=3, out_channels=32, kernel_size=9, stride=1),
        #     nn.InstanceNorm2d(32, affine=True),
        #     nn.ReLU(),
        #     ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=2),
        #     nn.InstanceNorm2d(64, affine=True),
        #     nn.ReLU(),
        #     ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=2),
        #     nn.InstanceNorm2d(128, affine=True),
        #     nn.ReLU(),
        #     ResidualBlock(128),
        #     ResidualBlock(128),
        #     ResidualBlock(128),
        #     ResidualBlock(128),
        #     ResidualBlock(128),
        #     UpsampleConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, upsample=2),
        #     nn.InstanceNorm2d(64, affine=True),
        #     nn.ReLU(),
        #     UpsampleConvLayer(in_channels=64, out_channels=32, kernel_size=3, stride=1, upsample=2),
        #     nn.InstanceNorm2d(32, affine=True),
        #     nn.ReLU(),
        #     ConvLayer(32, 3, kernel_size=9, stride=1)
        # )

    def forward(self, x):
        return self.net(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.refletion_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        """
        out shape:
            out = (in + 2 * padding_size - kernel_size) / stride + 1
        """
        out = self.refletion_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    """ResidualBlock
    from the paper: https://arxiv.org/abs/1512.03385
    InsranceNorm2d usually used in neural style transform
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvLayer(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(),
            ConvLayer(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True)
        )

    def forward(self, x):
        """keep the dimension"""
        residual = x
        out = self.block(x)
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    """
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReplicationPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


## Vgg16
class StyleVgg16(nn.Module):
    def __init__(self, requires_grad=True):
        super(StyleVgg16, self).__init__()
        features = vgg16(pretrained=True).features
        self.blk1 = nn.Sequential()
        self.blk2 = nn.Sequential()
        self.blk3 = nn.Sequential()
        self.blk4 = nn.Sequential()
        for i in range(4):
            self.blk1.add_module(str(i), features[i])
        for i in range(4, 9):
            self.blk2.add_module(str(i), features[i])
        for i in range(9, 16):
            self.blk3.add_module(str(i), features[i])
        for i in range(16, 23):
            self.blk4.add_module(str(i), features[i])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h1 = self.blk1(x)
        h2 = self.blk2(h1)
        h3 = self.blk3(h2)
        h4 = self.blk4(h3)
        vgg_outputs = namedtuple("Outputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h1, h2, h3, h4)
        return out


if __name__ == "__main__":
    x = torch.rand(1, 3, 256, 256)
    ## Test ConvLayer
    # conv = ConvLayer(1, 1, 2, 2)
    # print(conv(x).shape)


    ## Test UpsampleConvLayer
    # net = UpsampleConvLayer(1, 1, 2, 2, upsample=2)
    # print(net(x).shape)


    ## Test ResidualBlock
    # net = ResidualBlock(channels=1)
    # print(net(x).shape)


    ## Test TransformerNet
    net = TransformerNet()
    print(net(x).shape)