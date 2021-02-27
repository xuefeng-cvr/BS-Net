import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata
import numpy as np
from metrics import AverageMeter, Result
from models import modules as modules, net as net, dilation_resnet as resnet
import torch.nn.functional as F
import argparse
import util
import sobel
import scipy.io as sio
import os

parser = argparse.ArgumentParser(description='BS-Net NYUDv2 testing')
parser.add_argument('--path', '--p', default="BSN_NYUD.pth.tar", type=str,help='results_root (default:BSN_NYUD.pth.tar)')
def define_model(pre_train=True):
    original_model = resnet.resnet50(pretrained=pre_train)
    Encoder = modules.E_resnet(original_model)
    model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    return model

def main():
    global args
    args = parser.parse_args()
    model = define_model(pre_train=False)
    cudnn.benchmark = True
    val_loader = loaddata.getTestingData(1)

    checkpoint = torch.load(args.path)
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.cuda()

    print("=> loaded model (epoch {})".format(checkpoint["epoch"]))
    model.eval()  # switch to evaluate mode
    validate(val_loader,model)
    validate_PRF(val_loader,model)
    validate_VP(val_loader,model)

def validate(val_loader, model):
    average_meter = AverageMeter()
    end = time.time()

    for i, sample_batched in enumerate(val_loader):
        data_time = time.time() - end
        input, target = sample_batched['image'], sample_batched['depth']
        target = target.cuda(async=True)
        input = input.cuda()
        #with torch.no_grad():
            # compute output
        input=torch.autograd.Variable(input, volatile=True)
        target=torch.autograd.Variable(target, volatile=True)

        end=time.time()
        pred=model(input)
        pred=torch.nn.functional.interpolate(pred, size=[target.size(2), target.size(3)], mode='bilinear',align_corners=True)
        gpu_time=time.time()-end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i+1) % 300 == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MSE={result.mse:.2f}({average.mse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))
    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'Delta2={average.delta2:.3f}\n'
        'Delta3={average.delta3:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

def validate_PRF(val_loader, model):
    for th in [0.25,0.5,1]:
        totalNumber = 0
        Ae = 0
        Pe = 0
        Re = 0
        Fe = 0
        for i, sample_batched in enumerate(val_loader):
            input, target = sample_batched['image'], sample_batched['depth']
            totalNumber = totalNumber + input.size(0)

            target = target.cuda(async=True)
            input = input.cuda()
            with torch.no_grad():
                pred = model(input)
                pred=torch.nn.functional.interpolate(pred, size=[target.size(2), target.size(3)], mode='bilinear',align_corners=True)
                depth_edge = edge_detection(target)
                output_edge = edge_detection(pred)

                edge1_valid = (depth_edge > th)
                edge2_valid = (output_edge > th)
                edge1_valid = np.array(edge1_valid.data.cpu().numpy(), dtype=np.uint8)
                edge2_valid = np.array(edge2_valid.data.cpu().numpy(), dtype=np.uint8)

                equal=edge1_valid==edge2_valid
                nvalid = np.sum(equal)
                A = nvalid / (target.size(2) * target.size(3))

                nvalid2 = np.sum(((edge1_valid + edge2_valid) == 2))
                P = nvalid2 / (np.sum(edge2_valid))
                R = nvalid2 / (np.sum(edge1_valid))
                F = (2 * P * R) / (P + R)

                Ae += A
                Pe += P
                Re += R
                Fe += F
        Av = Ae / totalNumber
        Pv = Pe / totalNumber
        Rv = Re / totalNumber
        Fv = Fe / totalNumber
        print(th,'###################')
        print('avgPV:', Pv)
        print('avgRV:', Rv)
        print('avgFV:', Fv,end="\n")

def validate_VP(val_loader, model):
    totalNumber = 0
    De_6 = 0
    De_12 = 0
    De_24 = 0
    for i, sample_batched in enumerate(val_loader):
        input, target = sample_batched['image'], sample_batched['depth']
        totalNumber = totalNumber + input.size(0)

        target = target.cuda(async=True)
        input = input.cuda()
        with torch.no_grad():
            pred = model(input)
            pred=torch.nn.functional.interpolate(pred, size=[target.size(2), target.size(3)], mode='bilinear',align_corners=True)
            pred_6=torch.nn.functional.adaptive_avg_pool2d(pred,(6,6))
            pred_12=torch.nn.functional.adaptive_avg_pool2d(pred,(12,12))
            pred_24=torch.nn.functional.adaptive_avg_pool2d(pred,(24,24))
            gt_6=torch.nn.functional.adaptive_avg_pool2d(target, (6,6))
            gt_12=torch.nn.functional.adaptive_avg_pool2d(target, (12,12))
            gt_24=torch.nn.functional.adaptive_avg_pool2d(target, (24,24))

            D6=vp_dis(pred_6,gt_6)/8.48
            D12=vp_dis(pred_12, gt_12)/16.97
            D24=vp_dis(pred_24, gt_24)/33.94

            De_6+=D6
            De_12+=D12
            De_24+=D24

    De_6 = De_6 / totalNumber
    De_12 = De_12 / totalNumber
    De_24 = De_24 / totalNumber
    print("###################")
    print('De_6:', De_6)
    print('De_12:', De_12)
    print('De_24:', De_24)

def vp_dis(pred,gt):
    pred=pred.squeeze().cpu().detach().numpy()
    gt=gt.squeeze().cpu().detach().numpy()
    pred_index=np.unravel_index(pred.argmax(), pred.shape)
    gt_index=np.unravel_index(gt.argmax(), gt.shape)
    return ((pred_index[0]-gt_index[0])**2+(pred_index[1]-gt_index[1])**2)**0.5

def edge_detection(depth):
    get_edge = sobel.Sobel().cuda()
    edge_xy = get_edge(depth)
    edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
        torch.pow(edge_xy[:, 1, :, :], 2)
    edge_sobel = torch.sqrt(edge_sobel)

    return edge_sobel

if __name__ == '__main__':
    main()