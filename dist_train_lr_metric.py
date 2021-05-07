# main.py
import torch
import argparse
import torch.distributed as dist
from torch import optim
from myimgfolder import TrainImageFolder
from torchvision import transforms
# from colornet import ColorNet
from rescolornet import ColorNet
from discriminator import Discriminator
import torch.nn as nn
from tqdm import tqdm
from utils import LabLoss

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--class_weight', type=float, default=0.003)
parser.add_argument('--resume_from', default=None, help='the checkpoint file to resume from')
parser.add_argument('--pretrained', default=None, help='pretrained resnet50 model file')
parser.add_argument('--data_dir', help='the dir to get training data')
parser.add_argument('--output_dir', help='the dir to save logs and checkpoints')
args = parser.parse_args()


def adjust_loss_weights(epoch, init_weight=1.0, decay=0.6):
    loss_weight = init_weight * decay**epoch

    return loss_weight
        

def train(train_loader, optimizer, model, gan_optimizer, gan, epochs, output_dir, class_weight, resume_from=None):
    model.train()
    gan.train()
    start_epoch = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])

    for epoch in tqdm(range(start_epoch, epochs)):
        train_epoch(model, output_dir, epoch, train_loader, optimizer, gan_optimizer, gan, class_weight)


def train_epoch(model, output_dir, epoch, train_loader, optimizer, gan_optimizer, gan, class_weight):
    BCE_loss = nn.BCELoss(reduction='none').cuda()
    criterion_mse = nn.MSELoss(reduction='none').cuda()
    criterion_class = nn.CrossEntropyLoss(reduction='none').cuda()

    for batch_idx, (data, class_tar, weights) in enumerate(train_loader):
        if weights.prod() == 0:
            print('image_with_error loaded!', flush=True)
        messagefile = open(f'{output_dir}/message.txt', 'a')
        original_img = data[0].unsqueeze(1).float()  # 在第一维增加一个维度
        img_ab = data[1].float()
        original_img = original_img.cuda()
        img_ab = img_ab.cuda()
        class_tar = class_tar.cuda()
        weights = weights.float().cuda()

        optimizer.zero_grad()
        gan_optimizer.zero_grad()
        output, class_pred = model(original_img.expand(-1, 3, -1, -1), original_img.expand(-1, 3, -1, -1))

        if epoch % 2 == 0:
            fake_ab = output.detach()
            L_imgs = torch.cat([original_img, original_img], dim=0)
            ab_imgs = torch.cat([img_ab, fake_ab], dim=0)
            loss_w = torch.cat([weights, weights])
            ids_shuffle = torch.randperm(loss_w.shape[0])
            _, score_real_fake = gan(L_imgs[ids_shuffle], ab_imgs[ids_shuffle])
            # _, score_fake = gan(original_img, fake_ab)
            y_real_fake = torch.ones_like(score_real_fake)
            y_real_fake[img_ab.shape[0]:] = 0.0
            y_real_fake = y_real_fake[ids_shuffle]
            # y_fake = torch.zeros_like(score_fake, device=score_fake.device)
            dis_loss = BCE_loss(score_real_fake, y_real_fake)  # + BCE_loss(score_real, y_real)
            # print(score_real_fake[:, 0], y_real_fake[:, 0], flush=True)
            dis_loss_w = adjust_loss_weights(epoch=epoch, init_weight=1.0, decay=0.6)
            dis_loss = dis_loss_w * (dis_loss * loss_w[ids_shuffle]).sum() / (loss_w[ids_shuffle].sum() + 1e-12)

            dis_loss.backward()
            gan_optimizer.step()
            feat_loss = torch.zeros(1)
        else:
            with torch.no_grad():
                feat_real, _ = gan(original_img, img_ab)
            feat_fake, score_fake = gan(original_img, output)
            y_real = torch.ones_like(score_fake)       # to fool the discriminator
            dis_loss = BCE_loss(score_fake, y_real)
            dis_loss = (dis_loss * weights).sum() / (weights.sum() + 1e-12)
            feat_loss = criterion_mse(feat_fake, feat_real).mean(-1)
            feat_loss = (feat_loss * weights).sum() / (weights.sum() + 1e-12)

        ab_loss = criterion_mse(output, img_ab).mean([1, 2, 3])
        class_loss = criterion_class(class_pred, class_tar)
        ab_loss = (ab_loss * weights).sum() / (weights.sum() + 1e-12)
        class_loss = (class_loss * weights).sum() / (weights.sum() + 1e-12)
        img_loss = ab_loss + class_weight * class_loss

        if epoch % 2 == 0:
            loss = img_loss
        else:
            loss = img_loss + 0.01 * feat_loss + 0.0001 * dis_loss
        loss.backward()
        optimizer.step()  # 更新所有参数
        
        if batch_idx % 50 == 0 and dist.get_rank() == 0:
            message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\timg_loss:%.9f\tdis_loss:%.9f\tfeat_loss:%.9f\n' % (
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), img_loss.item(), dis_loss.item(), feat_loss.item())
            messagefile.write(message)
            torch.save(model.state_dict(), f'{output_dir}/colornet_params.pkl')
            print(message)
        messagefile.close()

    if epoch % 2 == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, f'{output_dir}/checkpoint_{epoch}.pth.tar')


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')

    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    print(f'local_rank: {local_rank}', flush=True)

    original_transform = transforms.Compose([
        transforms.Resize(256),  # 将输入的`PIL.Image`重新改变大小size，size是最小边的边长
        # 目前已经被transforms.Resize类取代了
        transforms.RandomCrop(224),  # 依据给定的size随机裁剪,在这种情况下，切出来的图片的形状是正方形
        transforms.RandomHorizontalFlip(),  # 随机水平翻转给定的PIL.Image,翻转的概率为0.5。
        # transforms.ToTensor()                              # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
    ])
    epochs = args.epochs
    data_dir = args.data_dir
    resume_from = args.resume_from
    output_dir = args.output_dir
    batch_size = args.batch_size
    lr = args.learning_rate
    class_weight = args.class_weight
    # 每个进程一个sampler
    model = ColorNet(pre_path=args.pretrained)
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    gan = Discriminator()
    gan.cuda(args.local_rank)
    gan = torch.nn.parallel.DistributedDataParallel(gan, device_ids=[args.local_rank], find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    gan_optimizer = optim.Adam(gan.parameters(), lr=lr) 
    print('==============preparing dataset============')
    train_dataset = TrainImageFolder(data_dir, original_transform)
    print('==============dataset ready============')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=4)
    train(train_loader, optimizer, model, gan_optimizer, gan, epochs, output_dir, class_weight, resume_from=None)
