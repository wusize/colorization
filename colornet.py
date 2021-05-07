import torch.nn as nn
import torch.nn.functional as F
import torch
from attention import SpatialAttention, ChannelAttention


class LowLevelFeatNet(nn.Module):
    def __init__(self):
        super(LowLevelFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

    def forward(self, x1, x2):
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = F.relu(self.bn4(self.conv4(x1)))
        x1 = F.relu(self.bn5(self.conv5(x1)))
        x1 = F.relu(self.bn6(self.conv6(x1)))
        if self.training:
            x2 = x1.clone()
        else:
            x2 = F.relu(self.bn1(self.conv1(x2)))
            x2 = F.relu(self.bn2(self.conv2(x2)))
            x2 = F.relu(self.bn3(self.conv3(x2)))
            x2 = F.relu(self.bn4(self.conv4(x2)))
            x2 = F.relu(self.bn5(self.conv5(x2)))
            x2 = F.relu(self.bn6(self.conv6(x2)))
        return x1, x2


class MidLevelFeatNet(nn.Module):
    def __init__(self):
        super(MidLevelFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class GlobalFeatNet(nn.Module):
    def __init__(self):
        super(GlobalFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(25088, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, 25088)
        x = F.relu(self.bn5(self.fc1(x)))
        output_512 = F.relu(self.bn6(self.fc2(x)))
        output_256 = F.relu(self.bn7(self.fc3(output_512)))
        return output_512, output_256


class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1000)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)

        return x


class ColorizationNet(nn.Module):
    def __init__(self, ch_att=False, sp_att=False, use_sigmoid=False):
        super(ColorizationNet, self).__init__()
        self.ch_att = ch_att
        self.sp_att = sp_att
        self.conv_pre = nn.Conv2d(256 + 256, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        if use_sigmoid:
            assert not ch_att
            assert not sp_att
        if self.ch_att:
            self.channel_attention = ChannelAttention(32)
        if self.sp_att:
            self.spatial_attention = SpatialAttention(32)
        self.use_sigmoid = use_sigmoid

        self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, mid_input, global_input):
        h = mid_input.shape[2]
        w = mid_input.shape[3]
        bs = global_input.shape[0]
        global_input = global_input.view(bs, -1, 1, 1)
        global_input = global_input.expand(-1, -1, h, w)  # batch*2048*w*h
        fusion_layer = torch.cat((mid_input, global_input), 1)  # batch*(256+2048)*w*h
        x = F.relu(self.bn1(self.conv_pre(fusion_layer)))
        x = F.relu(self.bn2(self.conv1(x)))
        x = self.upsample(x)
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn4(self.conv3(x)))
        x = self.upsample(x)
        if self.use_sigmoid:
            x = F.sigmoid(self.bn5(self.conv4(x)))
        else:
            x = F.relu(self.bn5(self.conv4(x)))
        if self.ch_att and self.sp_att:
            ch_att = self.channel_attention(x)
            sp_att = self.spatial_attention(x)
            att = ch_att.mul(sp_att)
            x = x + x.mul(att)
        elif self.ch_att and not self.sp_att:
            ch_att = self.channel_attention(x)
            x = x + x.mul(ch_att)
        elif not self.ch_att and self.sp_att:
            sp_att = self.spatial_attention(x)
            x = x + x.mul(sp_att)

        x = self.upsample(self.conv5(x))
        return x


class ColorNet(nn.Module):
    def __init__(self, ch_att=False, sp_att=False, use_sigmoid=False):
        super(ColorNet, self).__init__()
        self.low_lv_feat_net = LowLevelFeatNet()
        self.mid_lv_feat_net = MidLevelFeatNet()
        self.global_feat_net = GlobalFeatNet()
        self.class_net = ClassificationNet()
        self.upsample_col_net = ColorizationNet(ch_att, sp_att, use_sigmoid=use_sigmoid)

    def forward(self, x1, x2):
        x1, x2 = self.low_lv_feat_net(x1, x2)
        x1 = self.mid_lv_feat_net(x1)
        class_input, x2 = self.global_feat_net(x2)
        class_output = self.class_net(class_input)
        output = self.upsample_col_net(x1, x2)

        return output, class_output
