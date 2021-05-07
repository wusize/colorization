import torch.nn as nn
import torch


class ConvBn(nn.Module):
    # define layers
    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, 
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        
    def forward(self, x):
        return self.bn(self.conv(x))


class Discriminator(nn.Module):
    # define layers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(ConvBn(3, 64, 5, 2, 2),
                                         nn.LeakyReLU(),
                                         ConvBn(64, 128, 5, 2, 2),
                                         nn.LeakyReLU(),
                                         ConvBn(128, 256, 5, 2, 2),
                                         nn.LeakyReLU(),
                                         ConvBn(256, 512, 3, 2, 1),
                                         nn.LeakyReLU())
        
        self.fc_layer1 = nn.Linear(14 * 14 * 512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.fc_layer2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        
    def discriminator(self, x):
        x1 = self.conv_layers(x)
        x2 = x1.view(-1, 14 * 14 * 512)
        feature = self.relu1(self.bn1(self.fc_layer1(x2)))
        x3 = self.fc_layer2(feature)
        score = self.sigmoid(x3)
        
        return feature, score

    def forward(self, L, ab):
        Lab = torch.cat([L, ab], 1)
        feature, score = self.discriminator(Lab)
        return feature, score
