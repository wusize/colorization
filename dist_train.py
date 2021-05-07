# main.py
import torch
import argparse
import torch.distributed as dist
from torch import optim
from myimgfolder import TrainImageFolder
from torchvision import transforms
# from colornet import ColorNet
from rescolornet import ColorNet
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
parser.add_argument('--channel_attention', default=False)
parser.add_argument('--spatial_attention', default=False)
parser.add_argument('--use_sigmoid', default=False)
parser.add_argument('--pretrained', default=None, help='pretrained resnet50 model file')
parser.add_argument('--data_dir', help='the dir to get training data')
parser.add_argument('--output_dir', help='the dir to save logs and checkpoints')
args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, base_lr):
    lr = base_lr * 0.95**epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

def train(train_loader, optimizer, model, epochs, output_dir, class_weight, resume_from=None):
    model.train()
    start_epoch = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])

    for epoch in tqdm(range(start_epoch, epochs)):
        train_epoch(model, output_dir, epoch, train_loader, optimizer, class_weight)


def train_epoch(model, output_dir, epoch, train_loader, optimizer, class_weight):
    criterion_color = nn.MSELoss(reduction='none').cuda()
    criterion_class = nn.CrossEntropyLoss(reduction='none').cuda()
    for batch_idx, (data, class_tar, weights) in enumerate(train_loader):
        if weights.prod() == 0:
            print('image_with_error loaded!', flush=True)
        messagefile = open(f'{output_dir}/message.txt', 'a')
        original_img = data[0].unsqueeze(1).float()  # 在第一维增加一个维度
        img_ab = data[1].float()
        original_img = original_img.cuda()
        img_ab = img_ab.cuda()
        weights = weights.float().cuda()
        assert img_ab.shape[-1] == original_img.shape[-1]
        assert img_ab.shape[-2] == original_img.shape[-2]
        class_tar = class_tar.cuda()
        optimizer.zero_grad()
        output, class_pred = model(original_img.expand(-1, 3, -1, -1), original_img.expand(-1, 3, -1, -1))
        # 前向传播求出预测的值
        # print(class_pred.shape, class_tar, flush=True)
        ab_loss = criterion_color(output, img_ab).mean([1, 2, 3])
        class_loss = criterion_class(class_pred, class_tar)
        ab_loss = (ab_loss * weights).sum() / (weights.sum() + 1e-12)
        class_loss = (class_loss * weights).sum() / (weights.sum() + 1e-12)
        loss = ab_loss + class_weight * class_loss
        loss.backward()
        optimizer.step()  # 更新所有参数
        if batch_idx % 100 == 0 and dist.get_rank() == 0:
            message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\t ab_loss:%.9f\t class_loss:%.9f\n' % (
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), ab_loss.item(), class_loss.item())
            messagefile.write(message)
            torch.save(model.state_dict(), f'{output_dir}/colornet_params.pkl')
            print(message, flush=True)
        messagefile.close()

    if epoch % 2 == 0 and dist.get_rank() == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, f'{output_dir}/checkpoint_{epoch}.pth.tar')


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    # torch.cuda.set_device(args.local_rank)

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
    pretrained = args.pretrained
    output_dir = args.output_dir
    batch_size = args.batch_size
    channel_attention = args.channel_attention
    spatial_attention = args.spatial_attention
    use_sigmoid = args.use_sigmoid
    lr = args.learning_rate
    class_weight = args.class_weight
    print(args, flush=True)
    # 每个进程一个sampler
    model = ColorNet(pre_path=pretrained, ch_att=channel_attention, sp_att=spatial_attention, use_sigmoid=use_sigmoid)
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('==============preparing dataset============', flush=True)
    train_dataset = TrainImageFolder(data_dir, original_transform)
    print('==============dataset ready============', flush=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=4)
    train(train_loader, optimizer, model, epochs, output_dir, class_weight, resume_from=resume_from)
