from PIL import Image
from torchvision import transforms
import numpy as np
import torch
from skimage.color import lab2rgb, rgb2gray
import matplotlib.pyplot as plt
from torch.autograd import Variable
from rescolornet import ColorNet
from utils.transform_models import load_with_module
import os

# 配置cuda
have_cuda = torch.cuda.is_available()
color_model = ColorNet(sp_att=True, ch_att=True)
color_model.load_state_dict(load_with_module('/home/wusize/work_dirs/colorization/models/'
                                             'resnet/pre_trained/output_all_in/colornet_params_4.pkl'))
# 参数路径
if have_cuda:
    color_model.cuda()
color_model.eval()


# 处理图像,转成黑白和彩色
def Picture(name):
    img_name = name  # 输入图片的路径
    base_name = os.path.basename(img_name)
    img = Image.open(img_name)
    scale_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),  # 剪切图片大小为224*224
    ])
    img1 = scale_transform(img)

    img_scale = np.asarray(img1)
    img_original = np.asarray(img)

    img_scale = rgb2gray(img_scale)
    img_scale = torch.from_numpy(img_scale)
    img_original = rgb2gray(img_original)
    img_original = torch.from_numpy(img_original)
    print(img_original.size())

    # 阔展成4维
    original_img = img_original.unsqueeze(0).float()
    original_img = original_img.unsqueeze(1).float()

    print(original_img.size())
    gray_name = '/home/wusize/work_dirs/colorization/result/bw/' + base_name  # 生成黑白图像的路径
    pic = original_img.squeeze().numpy()
    pic = pic.astype(np.float64)
    plt.imsave(gray_name, pic, cmap='gray')
    print(original_img.size())
    w = original_img.size()[2]
    h = original_img.size()[3]

    scale_img = img_scale.unsqueeze(0).float()
    scale_img = scale_img.unsqueeze(1).float()

    if have_cuda:
        original_img, scale_img = original_img.cuda(), scale_img.cuda()
    with torch.no_grad():
        original_img, scale_img = Variable(original_img), Variable(scale_img)
    # 输入网络，scale_image进入全局特征提取的网络，若要做风格迁移，则scale_image改成另一个图片
    c_output, _ = color_model(original_img.expand(-1, 3, -1, -1), scale_img.expand(-1, 3, -1, -1))
    output, sp_att_plane = c_output
    print(original_img.shape, output.shape)
    color_img = torch.cat((original_img, output[:,:,:w,:h]), 1)  # L与ab融合
    print(color_img.size())
    color_img = color_img.data.cpu().numpy().transpose((0, 2, 3, 1))  # 转置

    print(type(color_img))

    color_img = color_img[0]
    color_img[:, :, 0:1] = color_img[:, :, 0:1] * 100
    color_img[:, :, 1:3] = color_img[:, :, 1:3] * 255 - 128
    color_img = color_img.astype(np.float64)
    color_img = lab2rgb(color_img)
    color_name = '/home/wusize/work_dirs/colorization/result/' + base_name  # 生成彩色图像的路径
    att_name = '/home/wusize/work_dirs/colorization/result/' + 'sp_att_plane' + base_name
    plt.imsave(color_name, color_img)
    print(sp_att_plane.shape)
    sp_att_plane = sp_att_plane[0, 0].detach().numpy()
    plt.imsave(att_name, sp_att_plane, cmap='viridis')
    ref = np.zeros((100, 10))
    ref2 = ref.copy()
    ref2[:50] = 1.0
    for i in range(100):
        ref[i] = float(i / 100)
    plt.imsave('/home/wusize/work_dirs/colorization/result/ref_0.jpg', ref, cmap='viridis')
    plt.imsave('/home/wusize/work_dirs/colorization/result/ref_1.jpg', ref2, cmap='binary')
    ab_vis = output[0, :, :w, :h].detach().numpy()
    plt.imsave('/home/wusize/work_dirs/colorization/result/a.jpg', ab_vis[0], cmap='RdYlGn_r')
    plt.imsave('/home/wusize/work_dirs/colorization/result/b.jpg', ab_vis[1], cmap='YlGnBu_r')

    return gray_name, color_name


def Picturestyle(name, namestyle):
    base_name = os.path.basename(name)
    img_name = name  # 输入图片的路径
    img = Image.open(img_name)
    scale_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),  # 剪切图片大小为224*224
    ])
    img1 = scale_transform(img)
    img_scale = np.asarray(img1)
    img_original = np.asarray(img)

    img_style = namestyle
    style_img = Image.open(img_style)
    scale_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),  # 剪切图片大小为224*224
    ])
    img2 = scale_transform(style_img)

    style = np.asarray(img2)
    style = rgb2gray(style)
    style = torch.from_numpy(style)
    style = style.unsqueeze(0).float()
    style = style.unsqueeze(1).float()

    img_scale = rgb2gray(img_scale)
    img_scale = torch.from_numpy(img_scale)
    img_original = rgb2gray(img_original)
    img_original = torch.from_numpy(img_original)
    print(img_original.size())

    # 阔展成4维
    original_img = img_original.unsqueeze(0).float()
    original_img = original_img.unsqueeze(1).float()

    print(original_img.size())
    gray_name = '/home/wusize/work_dirs/colorization/result/' + 'gray1' + '.jpg'  # 生成黑白图像的路径
    pic = original_img.squeeze().numpy()
    pic = pic.astype(np.float64)
    plt.imsave(gray_name, pic, cmap='gray')
    print(original_img.size())
    w = original_img.size()[2]
    h = original_img.size()[3]

    scale_img = img_scale.unsqueeze(0).float()
    scale_img = scale_img.unsqueeze(1).float()

    if have_cuda:
        original_img, scale_img, style = original_img.cuda(), scale_img.cuda(), style.cuda()
    with torch.no_grad():
        original_img, scale_img, style = Variable(original_img), Variable(scale_img),Variable(style)
    # 输入网络，scale_image进入全局特征提取的网络，若要做风格迁移，则scale_image改成另一个图片
    c_output, _ = color_model(original_img.expand(-1, 3, -1, -1), style.expand(-1, 3, -1, -1))
    output, sp_att_plane = c_output

    color_img = torch.cat((original_img, output[:,:,:w,:h]), 1)  # L与ab融合
    print(color_img.size())
    color_img = color_img.data.cpu().numpy().transpose((0, 2, 3, 1))  # 转置

    print(type(color_img))

    color_img = color_img[0]
    color_img[:, :, 0:1] = color_img[:, :, 0:1] * 100
    color_img[:, :, 1:3] = color_img[:, :, 1:3] * 255 - 128
    color_img = color_img.astype(np.float64)
    color_img = lab2rgb(color_img)
    color_name = '/home/wusize/work_dirs/colorization/result/' + base_name  # 生成彩色图像的路径
    plt.imsave(color_name, color_img)

    return color_name



