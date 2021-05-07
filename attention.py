import torch.nn as nn
import torch

class conv_bn_relu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, has_relu=True, groups=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, groups=groups)
        self.has_relu = has_relu
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x
    
class Attention(nn.Module):

    def __init__(self, output_chl_num):
        super(Attention, self).__init__()
        self.output_chl_num = output_chl_num
        self.conv_bn_relu_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                stride=1, padding=0, has_relu=True)
        self.conv_bn_relu_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=9,
                stride=1, padding=4, has_relu=False, groups=self.output_chl_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.conv_bn_relu_1(x)
        x = self.conv_bn_relu_2(x)
        x = self.sigmoid(x)
        out = input.mul(x) + input
        
        return out


class ChannelAttention(nn.Module):
    def __init__(self, output_chl_num):
        super(ChannelAttention, self).__init__()
        self.output_chl_num = output_chl_num
        self.conv_bn_relu_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                stride=1, padding=0, has_relu=True)
        self.conv_bn_relu_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                stride=1, padding=0, has_relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_bn_relu_1(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = self.conv_bn_relu_2(x)
        x = self.sigmoid(x)
    
        return x


class SpatialAttention(nn.Module):
    def __init__(self, in_chl_num):
        super(SpatialAttention, self).__init__()
        self.in_chl_num = in_chl_num
        self.conv_bn_relu_1 = conv_bn_relu(self.in_chl_num, self.in_chl_num, kernel_size=1,
                stride=1, padding=0, has_relu=True)
        self.conv_bn_relu_2 = conv_bn_relu(self.in_chl_num, 1, kernel_size=9,
                stride=1, padding=4, has_relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_bn_relu_1(x)
        x = self.conv_bn_relu_2(x)
        x = self.sigmoid(x)
        
        return x