import math
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from myimgfolder import TrainImageFolder
import torch.utils.model_zoo as model_zoo
BatchNorm2d = nn.BatchNorm2d

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv1 = conv3x3(3, 64, stride=2)
        # self.bn1 = BatchNorm2d(64)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(64, 64)
        # self.bn2 = BatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = conv3x3(64, 128)
        # self.bn3 = BatchNorm2d(128)
        # self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def get_lower_feature(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu2(self.bn2(self.conv2(x)))
        # x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        return x

    def forward(self, x):
        x = self.get_lower_feature(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feature_1d = x.view(x.size(0), -1)
        one_hot = self.fc(feature_1d)

        return feature_1d, one_hot


def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url, model_dir=model_dir)


def resnet50(model_path=None):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_path is not None:
        print(f'========loaf pretrained model from: {model_path}', flush=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print('no pretrained model', flush=True)
    return model


if __name__ == '__main__':
    from torchsummary import summary
    original_transform = transforms.Compose([
        transforms.Resize(256),  # 将输入的`PIL.Image`重新改变大小size，size是最小边的边长
        # 目前已经被transforms.Resize类取代了
        transforms.RandomCrop(224),  # 依据给定的size随机裁剪,在这种情况下，切出来的图片的形状是正方形
        transforms.RandomHorizontalFlip(),  # 随机水平翻转给定的PIL.Image,翻转的概率为0.5。
        # transforms.ToTensor()                              # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
    ])
    model_path = './resnet_backbone.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = resnet50().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    # summary(net, (3, 224, 224))
    train_set = TrainImageFolder("F:/dataset/jpg/", original_transform)  # 建训练集
    train_set_size = len(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    for batch_idx, (data, classes) in enumerate(train_loader):
        print(batch_idx, ' Class', classes)
        original_img = data[0].float().to(device)  # 在第一维增加一个维度
        img_ab = data[1].float().to(device)
        print(original_img.shape, img_ab.shape)
        res = net(original_img)
        print(res.shape)



