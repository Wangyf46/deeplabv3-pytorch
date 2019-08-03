import argparse
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import ipdb
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms
from tensorboardX import SummaryWriter

import deeplab
from pascal import VOCSegmentation
from cityscapes import Cityscapes
from utils import AverageMeter, inter_and_union

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--gpu', type=str, default='6',
                    help='test time gpu device id')
parser.add_argument('--backbone', type=str, default='resnet101',
                    help='resnet101')
parser.add_argument('--dataset', type=str, default='pascal',
                    help='pascal or cityscapes')
parser.add_argument('--groups', type=int, default=None, 
                    help='num of groups for group normalization')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.007,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--weight_std', action='store_true', default=False,
                    help='weight standardization')
parser.add_argument('--beta', action='store_true', default=False,
                    help='resnet101 beta')
parser.add_argument('--crop_size', type=int, default=513,
                    help='image crop size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
log_dir = os.path.join('ckpt', str(datetime.datetime.now()))
writer = SummaryWriter(log_dir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

def main():
  assert torch.cuda.is_available()
  torch.backends.cudnn.benchmark = True
  model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(args.backbone, args.dataset, args.exp)
  if args.dataset == 'pascal':
    dataset = VOCSegmentation('/data/wangyf/datasets/VOC2012/',
        train=args.train, crop_size=args.crop_size)
  elif args.dataset == 'cityscapes':
    dataset = Cityscapes('data/cityscapes',
        train=args.train, crop_size=args.crop_size)
  else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))
  if args.backbone == 'resnet101':
    model = getattr(deeplab, 'resnet101')(
        pretrained=(not args.scratch),
        num_classes=len(dataset.CLASSES),
        num_groups=args.groups,
        weight_std=args.weight_std,
        beta=args.beta)
  else:
    raise ValueError('Unknown backbone: {}'.format(args.backbone))

  if args.train:
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    model = nn.DataParallel(model).cuda()
    model.train()
    if args.freeze_bn:
      for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
          m.eval()
          m.weight.requires_grad = False
          m.bias.requires_grad = False
    backbone_params = (
        list(model.module.conv1.parameters()) +
        list(model.module.bn1.parameters()) +
        list(model.module.layer1.parameters()) +
        list(model.module.layer2.parameters()) +
        list(model.module.layer3.parameters()) +
        list(model.module.layer4.parameters()))
    last_params = list(model.module.aspp.parameters())
    optimizer = optim.SGD([
      {'params': filter(lambda p: p.requires_grad, backbone_params)},
      {'params': filter(lambda p: p.requires_grad, last_params)}],
      lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.train,
                                                 pin_memory=True, num_workers=args.workers)
    max_iter = args.epochs * len(dataset_loader)
    print(max_iter)
    losses = AverageMeter()
    start_epoch = 0

    if args.resume:
      if os.path.isfile(args.resume):
        print('=> loading checkpoint {0}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> loaded checkpoint {0} (epoch {1})'.format(
          args.resume, checkpoint['epoch']))
      else:
        print('=> no checkpoint found at {0}'.format(args.resume))

    print(start_epoch)
    for epoch in range(start_epoch, args.epochs):
      for i, (inputs, target, _, _) in enumerate(dataset_loader):
        cur_iter = epoch * len(dataset_loader) + i
        lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * args.last_mult

        inputs = Variable(inputs.cuda())
        target = Variable(target.cuda())
        outputs = model(inputs)
        loss = criterion(outputs, target)
        if np.isnan(loss.item()) or np.isinf(loss.item()):
          ipdb.set_trace()
        losses.update(loss.item(), args.batch_size)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('epoch: {0}\t'
              'iter: {1}/{2}\t'
              'lr: {3:.6f}\t'
              'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
              epoch + 1, i + 1, len(dataset_loader), lr, loss=losses))

      writer.add_scalar('loss_ema', losses.ema, epoch)
      writer.add_scalar('lr', lr, epoch)
      if epoch % 10 == 9:
        torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          }, model_fname % (epoch + 1))

  else:
    model = model.cuda()
    model.eval()
    #checkpoint = torch.load(model_fname % args.epochs)
    checkpoint = torch.load('data/source_model/deeplab_resnet101_pascal_v3_bn_lr7e-3_epoch50.pth')
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model.load_state_dict(state_dict)
    cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']  # (256, 3)
    cmap = (cmap * 255).astype(np.uint8).flatten().tolist() # 768

    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    for i in range(len(dataset)):
      inputs, target, w, h = dataset[i]   # torch.Size([3,513,513]), torch.Size([513, 513])
      inputs = Variable(inputs.cuda())
      outputs = model(inputs.unsqueeze(0)) # ([1,3,513,513])->([1,21,513,513])
      _, pred = torch.max(outputs, 1)  # ([1,513,513])
      pred = pred.data.cpu().numpy().squeeze().astype(np.uint8) #(513, 513)
      mask = target.numpy().astype(np.uint8)
      mask = Image.fromarray(mask).crop((0,0,w,h))
      plt.subplot(121)
      plt.imshow(mask)
      plt.axis('off')
      imname = dataset.masks[i].split('/')[-1]
      mask_pred = Image.fromarray(pred)
      mask_pred.putpalette(cmap)
      mask_pred = mask_pred.crop((0, 0, w, h))
      #mask_pred.save(os.path.join('data/val', imname))
      plt.subplot(122)
      plt.imshow(mask_pred)
      plt.axis('off')
      plt.savefig(os.path.join('data/val', imname))
      #plt.show()
      print('eval: {0}/{1}'.format(i + 1, len(dataset)))

      inter, union = inter_and_union(mask_pred, mask, len(dataset.CLASSES)) ## TODO
      inter_meter.update(inter)
      union_meter.update(union)

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    for i, val in enumerate(iou):
      print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
    print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))


if __name__ == "__main__":
  main()
