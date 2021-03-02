import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata
import sobel
import os
import argparse

from models import modules as modules, net as net, dilation_resnet as resnet

from util import compute_global_errors,\
                 compute_directed_depth_error,\
                 compute_depth_boundary_error,\
                 compute_planarity_error,\
                 compute_distance_related_errors

parser = argparse.ArgumentParser(description='BS-Net iBims-1 testing')
parser.add_argument('--path', '--p', default="BSN_NYUD.pth.tar", type=str,help='results_root (default:BSN_NYUD.pth.tar)')

os.environ['CUDA_VISIBLE_DEVICES']='1'
with open('./data/iBims1/imagelist.txt') as f:
    image_names = f.readlines()
image_names = [x.strip() for x in image_names]

num_samples = len(image_names) # number of images

# Initialize global and geometric errors ...
rms     = np.zeros(num_samples, np.float32)
log10   = np.zeros(num_samples, np.float32)
abs_rel = np.zeros(num_samples, np.float32)
sq_rel  = np.zeros(num_samples, np.float32)
thr1    = np.zeros(num_samples, np.float32)
thr2    = np.zeros(num_samples, np.float32)
thr3    = np.zeros(num_samples, np.float32)

abs_rel_vec = np.zeros((num_samples,20),np.float32)
log10_vec = np.zeros((num_samples,20),np.float32)
rms_vec = np.zeros((num_samples,20),np.float32)

dde_0   = np.zeros(num_samples, np.float32)
dde_m   = np.zeros(num_samples, np.float32)
dde_p   = np.zeros(num_samples, np.float32)

dbe_acc = np.zeros(num_samples, np.float32)
dbe_com = np.zeros(num_samples, np.float32)

pe_fla = np.empty(0)
pe_ori = np.empty(0)

def define_model(pre_train=True):
    original_model = resnet.resnet50(pretrained=pre_train)
    Encoder = modules.E_resnet(original_model)
    model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    return model

def main():
    model = define_model(pre_train=False)
    cudnn.benchmark = True
    global args
    args=parser.parse_args()
    val_loader = loaddata.getTestingData_iBims1(1)

    checkpoint = torch.load(args.path)
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)
    model.cuda()
    model.eval() # switch to evaluate mode
    print("=> loaded model (epoch {})".format(checkpoint["epoch"]))

    validate(val_loader, model)
    validate_PRF(val_loader,model)
    validate_VP(val_loader,model)



def validate(val_loader, model):
    for i, sample_batched in enumerate(val_loader):
        #print('正在处理：{0}'.format(i))
        input, target, edges, calib, mask_invalid, mask_transp, mask_wall, \
        paras_wall, mask_table, paras_table, mask_floor, paras_floor=sample_batched['image'], sample_batched['depth'], \
                                                                    sample_batched['edges'], sample_batched['calib'], \
                                                                    sample_batched['mask_invalid'], sample_batched['mask_transp'], \
                                                                    sample_batched['mask_wall'], sample_batched['mask_wall_paras'], \
                                                                    sample_batched['mask_table'], sample_batched['mask_table_paras'], \
                                                                    sample_batched['mask_floor'], sample_batched['mask_floor_paras']

        with torch.no_grad():
            input = torch.autograd.Variable(input)
            input = input.cuda()
            pred = model(input)
            pred=torch.nn.functional.interpolate(pred, size=[target.size(2), target.size(3)], mode='bilinear',align_corners=True)

            pred=pred.data[0].cpu().numpy().squeeze()
            depth=target.cpu().numpy().squeeze()

            edges=edges.numpy().squeeze()
            calib=calib.numpy().squeeze()
            mask_transp=mask_transp.numpy().squeeze()
            mask_invalid=mask_invalid.numpy().squeeze()
            mask_wall=mask_wall.numpy().squeeze()
            paras_wall=paras_wall.numpy().squeeze()
            mask_table=mask_table.numpy().squeeze()
            paras_table=paras_table.numpy().squeeze()
            mask_floor=mask_floor.numpy().squeeze()
            paras_floor=paras_floor.numpy().squeeze()

            pred[np.isnan(pred)] = 0
            pred_invalid = pred.copy()
            pred_invalid[pred_invalid != 0] = 1

            mask_missing = depth.copy()  # Mask for further missing depth values in depth map
            mask_missing[mask_missing != 0] = 1

            mask_valid = mask_transp * mask_invalid * mask_missing * pred_invalid  # Combine masks
            # Apply 'valid_mask' to raw depth map
            depth_valid = depth * mask_valid

            gt = depth_valid
            gt_vec = gt.flatten()

            # Apply 'valid_mask' to raw depth map
            pred = pred * mask_valid
            pred_vec = pred.flatten()
            # Compute errors ...
            abs_rel[i], sq_rel[i], rms[i], log10[i], thr1[i], thr2[i], thr3[i] = compute_global_errors(gt_vec, pred_vec)
            abs_rel_vec[i, :], log10_vec[i, :], rms_vec[i, :] = compute_distance_related_errors(gt, pred)
            dde_0[i], dde_m[i], dde_p[i] = compute_directed_depth_error(gt_vec, pred_vec, 3.0)
            dbe_acc[i], dbe_com[i], est_edges = compute_depth_boundary_error(edges, pred)

            mask_wall = mask_wall * mask_valid
            global pe_fla,pe_ori

            if paras_wall.size > 0:
                pe_fla_wall, pe_ori_wall = compute_planarity_error(gt, pred, paras_wall, mask_wall, calib)
                pe_fla = np.append(pe_fla, pe_fla_wall)
                pe_ori = np.append(pe_ori, pe_ori_wall)

            mask_table =mask_table * mask_valid
            if paras_table.size > 0:
                pe_fla_table, pe_ori_table = compute_planarity_error(gt, pred, paras_table, mask_table, calib)
                pe_fla = np.append(pe_fla, pe_fla_table)
                pe_ori = np.append(pe_ori, pe_ori_table)

            mask_floor = mask_floor * mask_valid
            if paras_floor.size > 0:
                pe_fla_floor, pe_ori_floor = compute_planarity_error(gt, pred, paras_floor, mask_floor, calib)
                pe_fla = np.append(pe_fla, pe_fla_floor)
                pe_ori = np.append(pe_ori, pe_ori_floor)

    print('Results:')
    print ('############ Global Error Metrics #################')
    print ('rel    = ', np.nanmean(abs_rel))
    print('sq_rel = ',  np.nanmean(sq_rel))
    print ('log10  = ', np.nanmean(log10))
    print ('rms    = ', np.nanmean(rms))
    print ('thr1   = ', np.nanmean(thr1))
    print ('thr2   = ', np.nanmean(thr2))
    print ('thr3   = ', np.nanmean(thr3))
    print ('############ Planarity Error Metrics #################')
    print('pe_fla = ', np.nanmean(pe_fla))
    print('pe_ori = ', np.nanmean(pe_ori))
    print ('############ Depth Boundary Error Metrics #################')
    print ('dbe_acc = ', np.nanmean(dbe_acc))
    print ('dbe_com = ', np.nanmean(dbe_com))
    print ('############ Directed Depth Error Metrics #################')
    print ('dde_0  = ', np.nanmean(dde_0) * 100.)
    print ('dde_m  = ', np.nanmean(dde_m) * 100.)
    print ('dde_p  = ', np.nanmean(dde_p) * 100.)

def validate_PRF(val_loader, model):

    for th in [0.25,0.5,1]:
        totalNumber = 0
        Ae = 0
        Pe = 0
        Re = 0
        Fe = 0
        for i, sample_batched in enumerate(val_loader):
            input, target, edges, calib, mask_invalid, mask_transp, mask_wall, \
            paras_wall, mask_table, paras_table, mask_floor, paras_floor=sample_batched['image'], sample_batched['depth'], \
                                                                         sample_batched['edges'], sample_batched['calib'], \
                                                                         sample_batched['mask_invalid'], sample_batched['mask_transp'], \
                                                                         sample_batched['mask_wall'], sample_batched['mask_wall_paras'], \
                                                                         sample_batched['mask_table'], sample_batched['mask_table_paras'], \
                                                                         sample_batched['mask_floor'], sample_batched['mask_floor_paras']
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