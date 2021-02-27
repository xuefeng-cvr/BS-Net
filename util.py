import torch
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from skimage import feature
from scipy import ndimage
from sklearn.decomposition import PCA
import math

cmap = plt.cm.viridis
def lg10(x):
    return torch.div(torch.log(x), math.log(10))

def maxOfTwo(x, y):
    z = x.clone()
    maskYLarger = torch.lt(x, y)
    z[maskYLarger.detach()] = y[maskYLarger.detach()]
    return z

def nValid(x):
    return torch.sum(torch.eq(x, x).float())

def nNanElement(x):
    return torch.sum(torch.ne(x, x).float())

def getNanMask(x):
    return torch.ne(x, x)

def setNanToZero(input, target):
    nanMask = getNanMask(target)
    nValidElement = nValid(target)

    _input = input.clone()
    _target = target.clone()

    _input[nanMask] = 0
    _target[nanMask] = 0

    return _input, _target, nanMask, nValidElement


def evaluateError(output, target):
    errors = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
              'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    _output, _target, nanMask, nValidElement = setNanToZero(output, target)

    if (nValidElement.data.cpu().numpy() > 0):
        diffMatrix = torch.abs(_output - _target)

        errors['MSE'] = torch.sum(torch.pow(diffMatrix, 2)) / nValidElement

        errors['MAE'] = torch.sum(diffMatrix) / nValidElement

        realMatrix = torch.div(diffMatrix, _target)
        realMatrix[nanMask] = 0
        errors['ABS_REL'] = torch.sum(realMatrix) / nValidElement

        LG10Matrix = torch.abs(lg10(_output) - lg10(_target))
        LG10Matrix[nanMask] = 0
        errors['LG10'] = torch.sum(LG10Matrix) / nValidElement

        yOverZ = torch.div(_output, _target)
        zOverY = torch.div(_target, _output)

        maxRatio = maxOfTwo(yOverZ, zOverY)

        errors['DELTA1'] = torch.sum(
            torch.le(maxRatio, 1.25).float()) / nValidElement
        errors['DELTA2'] = torch.sum(
            torch.le(maxRatio, math.pow(1.25, 2)).float()) / nValidElement
        errors['DELTA3'] = torch.sum(
            torch.le(maxRatio, math.pow(1.25, 3)).float()) / nValidElement

        errors['MSE'] = float(errors['MSE'].data.cpu().numpy())
        errors['ABS_REL'] = float(errors['ABS_REL'].data.cpu().numpy())
        errors['LG10'] = float(errors['LG10'].data.cpu().numpy())
        errors['MAE'] = float(errors['MAE'].data.cpu().numpy())
        errors['DELTA1'] = float(errors['DELTA1'].data.cpu().numpy())
        errors['DELTA2'] = float(errors['DELTA2'].data.cpu().numpy())
        errors['DELTA3'] = float(errors['DELTA3'].data.cpu().numpy())

    return errors


def addErrors(errorSum, errors, batchSize):
    errorSum['MSE']=errorSum['MSE'] + errors['MSE'] * batchSize
    errorSum['ABS_REL']=errorSum['ABS_REL'] + errors['ABS_REL'] * batchSize
    errorSum['LG10']=errorSum['LG10'] + errors['LG10'] * batchSize
    errorSum['MAE']=errorSum['MAE'] + errors['MAE'] * batchSize

    errorSum['DELTA1']=errorSum['DELTA1'] + errors['DELTA1'] * batchSize
    errorSum['DELTA2']=errorSum['DELTA2'] + errors['DELTA2'] * batchSize
    errorSum['DELTA3']=errorSum['DELTA3'] + errors['DELTA3'] * batchSize

    return errorSum


def averageErrors(errorSum, N):
    averageError={'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                    'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    averageError['MSE'] = errorSum['MSE'] / N
    averageError['ABS_REL'] = errorSum['ABS_REL'] / N
    averageError['LG10'] = errorSum['LG10'] / N
    averageError['MAE'] = errorSum['MAE'] / N

    averageError['DELTA1'] = errorSum['DELTA1'] / N
    averageError['DELTA2'] = errorSum['DELTA2'] / N
    averageError['DELTA3'] = errorSum['DELTA3'] / N

    return averageError


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min=np.min(depth)
    if d_max is None:
        d_max=np.max(depth)
    depth_relative=(depth-d_min)/(d_max-d_min)
    return 255*cmap(depth_relative)[:, :, :3]  # H, W, C


def merge_into_row(input, depth_target, depth_pred,object_mask,object_nums):
    rgb=np.transpose(np.squeeze(input), (2, 1, 0))  # H, W, C
    depth_target_cpu=np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu=np.squeeze(depth_pred.data.cpu().numpy())

    mask=object_mask==object_nums
    target_mse=depth_target_cpu[mask].mean()
    pred_mse=depth_pred_cpu[mask].mean()

    print(target_mse,pred_mse)

    indexs=np.argwhere(object_mask==object_nums)
    print(indexs.shape)
    min_x=np.min(indexs[:,0])
    min_y=np.min(indexs[:,1])

    max_x=np.max(indexs[:,0])
    max_y=np.max(indexs[:,1])
    print(min_x,min_y)
    print(max_x,max_y)



    d_min=min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max=max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col=colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col=colored_depthmap(depth_pred_cpu, d_min, d_max)

    depth_target_col=Image.fromarray(depth_target_col.astype('uint8'))
    depth_pred_col=Image.fromarray(depth_pred_col.astype('uint8'))

    font=ImageFont.truetype('LiberationSans-Regular.ttf', 35)
    draw=ImageDraw.Draw(depth_target_col)
    draw.rectangle((min_y, min_x, max_y, max_x), fill=None, outline='red')
    draw.text((min_y, min_x-50), str(target_mse)[0:3], font=font,fill=(255, 0, 0))
    draw=ImageDraw.Draw(depth_pred_col)
    draw.rectangle((min_y,min_x, max_y, max_x), fill=None, outline='red')
    draw.text((min_y, min_x-50), str(pred_mse)[0:3], font=font,fill=(255, 0, 0))

    depth_target_col=np.array(depth_target_col)
    depth_pred_col=np.array(depth_pred_col)

    img_merge=np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb=255*np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu=np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu=np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu=np.squeeze(depth_pred.data.cpu().numpy())

    d_min=min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max=max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col=colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col=colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col=colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge=np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge=Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv=nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        # edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_kx=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        edge_ky=np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k=np.stack((edge_kx, edge_ky))

        edge_k=torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight=nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad=False

    def forward(self, x):
        out=self.edge_conv(x)
        out=out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out

def compute_distance_related_errors(gt, pred):
    # initialize output
    abs_rel_vec_tmp = np.zeros(20, np.float32)
    log10_vec_tmp = np.zeros(20, np.float32)
    rms_vec_tmp = np.zeros(20, np.float32)

    # exclude masked invalid and missing measurements
    gt = gt[gt != 0]
    pred = pred[pred != 0]

    gt_all = gt
    pred_all = pred
    bot = 0.0
    idx = 0
    for top in range(1, 21):
        mask = np.logical_and(gt_all >= bot, gt_all <= top)
        gt_tmp = gt_all[mask]
        pred_tmp = pred_all[mask]
        # calc errors
        abs_rel_vec_tmp[idx], tmp, rms_vec_tmp[idx], log10_vec_tmp[idx], tmp, tmp, tmp = compute_global_errors(gt_tmp,
                                                                                                               pred_tmp)

        bot = top  # re-assign bottom threshold
        idx = idx + 1

    return abs_rel_vec_tmp, log10_vec_tmp, rms_vec_tmp


def compute_global_errors(gt, pred):
    # exclude masked invalid and missing measurements
    gt = gt[gt != 0]
    pred = pred[pred != 0]

    # compute global relative errors
    thresh = np.maximum((gt / pred), (pred / gt))
    thr1 = (thresh < 1.25).mean()
    thr2 = (thresh < 1.25 ** 2).mean()
    thr3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    log10 = np.mean(np.abs(np.log10(gt) - np.log10(pred)))

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, log10, thr1, thr2, thr3


def compute_directed_depth_error(gt, pred, thr):
    # exclude masked invalid and missing measurements
    gt = gt[gt != 0]
    pred = pred[pred != 0]

    # number of valid depth values
    nPx = float(len(gt))

    gt[gt <= thr] = 1  # assign depths closer as 'thr' as '1s'
    gt[gt > thr] = 0  # assign depths farer as 'thr' as '0s'
    pred[pred <= thr] = 1
    pred[pred > thr] = 0

    diff = pred - gt  # compute difference map

    dde_0 = np.sum(diff == 0) / nPx
    dde_m = np.sum(diff == 1) / nPx
    dde_p = np.sum(diff == -1) / nPx

    return dde_0, dde_m, dde_p


def compute_depth_boundary_error(edges_gt, pred):
    # skip dbe if there is no ground truth distinct edge
    if np.sum(edges_gt) == 0:
        dbe_acc = np.nan
        dbe_com = np.nan
        edges_est = np.empty(pred.shape).astype(int)
    else:

        # normalize est depth map from 0 to 1
        pred_normalized = pred.copy().astype('f')
        pred_normalized[pred_normalized == 0] = np.nan
        pred_normalized = pred_normalized - np.nanmin(pred_normalized)
        pred_normalized = pred_normalized / np.nanmax(pred_normalized)

        # apply canny filter
        edges_est = feature.canny(pred_normalized, sigma=np.sqrt(2), low_threshold=0.15, high_threshold=0.3)

        # compute distance transform for chamfer metric
        D_gt = ndimage.distance_transform_edt(1 - edges_gt)
        D_est = ndimage.distance_transform_edt(1 - edges_est)

        max_dist_thr = 10.  # Threshold for local neighborhood

        mask_D_gt = D_gt < max_dist_thr  # truncate distance transform map

        E_fin_est_filt = edges_est * mask_D_gt  # compute shortest distance for all predicted edges

        if np.sum(E_fin_est_filt) == 0:  # assign MAX value if no edges could be detected in prediction
            dbe_acc = max_dist_thr
            dbe_com = max_dist_thr
        else:
            # accuracy: directed chamfer distance of predicted edges towards gt edges
            dbe_acc = np.nansum(D_gt * E_fin_est_filt) / np.nansum(E_fin_est_filt)

            # completeness: sum of undirected chamfer distances of predicted and gt edges
            ch1 = D_gt * edges_est  # dist(predicted,gt)
            ch1[ch1 > max_dist_thr] = max_dist_thr  # truncate distances
            ch2 = D_est * edges_gt  # dist(gt, predicted)
            ch2[ch2 > max_dist_thr] = max_dist_thr  # truncate distances
            res = ch1 + ch2  # summed distances
            dbe_com = np.nansum(res) / (np.nansum(edges_est) + np.nansum(edges_gt))  # normalized

    return dbe_acc, dbe_com, edges_est


def compute_planarity_error(gt, pred, paras, mask, calib):
    # mask invalid and missing depth values
    pred[pred == 0] = np.nan
    gt[gt == 0] = np.nan

    # number of planes of the current plane type
    if(paras.ndim==1):
        paras=np.expand_dims(paras, 0);
    nr_planes = paras.shape[0]

    # initialize PE errors
    pe_fla = np.empty(0)
    pe_ori = np.empty(0)

    for j in range(nr_planes):  # loop over number of planes

        # only consider depth values for this specific planar mask
        curr_plane_mask = mask.copy()
        curr_plane_mask[curr_plane_mask < (j + 1)] = 0
        curr_plane_mask[curr_plane_mask > (j + 1)] = 0
        remain_mask = curr_plane_mask.astype(float)
        remain_mask[remain_mask == 0] = np.nan
        remain_mask[np.isnan(remain_mask) == 0] = 1

        # only consider plane masks which are bigger than 5% of the image dimension
        if np.nansum(remain_mask) / (640. * 480.) < 0.05:
            flat = np.nan
            orie = np.nan
        else:
            # scale remaining depth map of current plane towards gt depth map
            mean_depth_est = np.nanmedian(pred * remain_mask)
            mean_depth_gt = np.nanmedian(gt * remain_mask)
            est_depth_scaled = pred / (mean_depth_est / mean_depth_gt) * remain_mask

            # project masked and scaled depth values to 3D points
            fx_d = calib[0, 0]
            fy_d = calib[1, 1]
            cx_d = calib[2, 0]
            cy_d = calib[2, 1]
            # c,r = np.meshgrid(range(gt.shape[1]),range(gt.shape[0]))
            c, r = np.meshgrid(range(1, gt.shape[1] + 1), range(1, gt.shape[0] + 1))
            tmp_x = ((c - cx_d) * est_depth_scaled / fx_d)
            tmp_y = est_depth_scaled
            tmp_z = (-(r - cy_d) * est_depth_scaled / fy_d)
            X = tmp_x.flatten()
            Y = tmp_y.flatten()
            Z = tmp_z.flatten()
            X = X[~np.isnan(X)]
            Y = Y[~np.isnan(Y)]
            Z = Z[~np.isnan(Z)]
            pointCloud = np.stack((X, Y, Z))

            # fit 3D plane to 3D points (normal, d)
            pca = PCA(n_components=3)
            pca.fit(pointCloud.T)
            normal = -pca.components_[2, :]
            point = np.mean(pointCloud, axis=1)
            d = -np.dot(normal, point);

            # PE_flat: deviation of fitted 3D plane
            flat = np.std(np.dot(pointCloud.T, normal.T) + d) * 100.

            n_gt = paras[j, 4:7]
            if np.dot(normal, n_gt) < 0:
                normal = -normal

            # PE_ori: 3D angle error between ground truth plane and normal vector of fitted plane
            orie = math.atan2(np.linalg.norm(np.cross(n_gt, normal)), np.dot(n_gt, normal)) * 180. / np.pi

        pe_fla = np.append(pe_fla, flat)  # append errors
        pe_ori = np.append(pe_ori, orie)

    return pe_fla, pe_ori