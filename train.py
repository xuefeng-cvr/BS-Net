# -*- coding: UTF-8 -*-
import argparse
import time
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata
import random
import numpy as np
import util
from models import modules as modules, net as net, dilation_resnet as resnet

parser = argparse.ArgumentParser(description='BS-Net training')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seed', '--rs', default=1024, type=int,
                    help='random seed (default: 0)')
parser.add_argument('--resume', '--r', default="", type=str,
                    help='resume_root (default:"")')
########################################################

def define_model(pre_train=True):
    original_model = resnet.resnet50(pretrained=pre_train)
    Encoder = modules.E_resnet(original_model)
    model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    return model

def main():
    global args
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    model = define_model(pre_train=True)

    ####################load pretrained model
    if args.resume!="":
        Checkpoint=torch.load(args.resume)
        state_dict = Checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of dataparallel
            if "PSP" in name:
                name=name.replace("PSP", "DCE")
            if "MFF" in name:
                name=name.replace("MFF", "BUBF")
            if "rrb" in name:
                name=name.replace("rrb", "lrb")
            if "lfc" in name:
                name=name.replace("lfc", "ssp")
            if "srm" in name:
                name=name.replace("srm", "SRM")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        args.start_epoch=Checkpoint["epoch"]+1
        print('parameter loaded successfully!!')

    if torch.cuda.device_count() == 8:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        batch_size = 64
    elif torch.cuda.device_count() == 4:
        model = torch.nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()
        batch_size = 16
    elif torch.cuda.device_count() == 2:
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        batch_size = 8
    else:
        model = model.cuda()
        batch_size = 4  # batch size

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    train_loader = loaddata.getTrainingData(batch_size)
    losses={}
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        loss=train(train_loader, model, optimizer, epoch)
        losses[str(epoch)]=loss
        save_checkpoint({"epoch": epoch, "state_dict": model.state_dict(),"loss_avg":loss},
                        filename='midCheckpoint_{}.pth.tar'.format(epoch))

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient =util.Sobel().cuda()

    end = time.time()
    for i, sample_batched in enumerate(train_loader):
        image, depth = sample_batched['image'], sample_batched['depth']

        depth = depth.cuda()
        image = image.cuda()
        image = torch.autograd.Variable(image)
        depth = torch.autograd.Variable(depth)

        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)
        optimizer.zero_grad()
        #pdb.set_trace()
        output = model(image)
        #pdb.set_trace()
        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()

        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

        loss = loss_depth + loss_normal + (loss_dx + loss_dy)
        losses.update(loss.data, image.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'
              .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))
    return losses.avg

# adjust the learning rate every 5 epochs
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# define a useful data structure
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# save the model parameters
def save_checkpoint(state, filename='res50.pth.tar'):
    torch.save(state, filename)

if __name__ == '__main__':
    main()