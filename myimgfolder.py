from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray
import torch
import numpy as np
import os
#import matplotlib.pyplot as plt


class TrainImageFolder(datasets.ImageFolder):
    # ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字
    def __getitem__(self, index):
        """
                Args:
                    index (int): Index

                Returns:
                    tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]                 # self.imgs = imgs= make_dataset(root, class_to_idx)
                                                        # imgs (list): List of (image path, class_index) tuples
        try:
            img = self.loader(path)
            assert self.transform is not None
            img_original = self.transform(img)
            img_original = np.asarray(img_original)  # 将结构数据转化为ndarray
            weight = 1.0
        except:
            print(f'image with error: {path}', flush=True)
            img_original = np.zeros((224, 224, 3))
            weight = 0.0


        img_lab = rgb2lab(img_original)  # 转换为lab空间
        img_lab = (img_lab + 128) / 255  # 范围变为0到1
        img_ab = img_lab[:, :, 1:3]  # 不懂
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))  # transpose转置，
        # (height,width, channel)=>(channel,height,width)
        img_original = rgb2gray(img_original)
        img_original = torch.from_numpy(img_original)
        # if self.transform is not None:                  # self.transform就是original_transform

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img_original, img_ab), target, weight


scale_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    # transforms.ToTensor()
])


class ValImageFolder(datasets.ImageFolder):

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        img_scale = img.copy()
        img_original = img
        img_scale = scale_transform(img_scale)

        img_scale = np.asarray(img_scale)
        img_original = np.asarray(img_original)

        img_scale = rgb2gray(img_scale)
        img_scale = torch.from_numpy(img_scale)
        img_original = rgb2gray(img_original)
        img_original = torch.from_numpy(img_original)
        return (img_original, img_scale), target
