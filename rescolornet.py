import torch.nn as nn
from resnet import resnet50
from colornet import MidLevelFeatNet, ColorizationNet


class ColorNet(nn.Module):
    def __init__(self, pre_path=None, ch_att=False, sp_att=False, use_sigmoid=False):
        super(ColorNet, self).__init__()
        self.resnet_backbone = resnet50(pre_path)
        self.mid_lv_feat_net = MidLevelFeatNet()
        self.upsample_col_net = ColorizationNet(ch_att, sp_att, use_sigmoid)
        self.down_sample = nn.Sequential(nn.Linear(2048, 256),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU())

    def forward(self, x1, x2):
        x1 = self.resnet_backbone.get_lower_feature(x1)
        x1 = self.mid_lv_feat_net(x1)
        x2, one_hot_class = self.resnet_backbone(x2)
        x2 = self.down_sample(x2)
        output = self.upsample_col_net(x1, x2)
        return output, one_hot_class
