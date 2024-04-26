import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
from chamfer_distance import ChamferDistance
from datasets.dhb_set import test_DHBDataset
from model_icme.SRNet import SRNet
from tqdm import tqdm
import argparse
from model_icme.main_utils import *
from collections import defaultdict
#from datasets.dhb_set import test_DHBDataset
import time


def normalize_point_cloud(input):
    """
    input: pc [B, N, 3]
    output: pc, centroid, furthest_distance
    """
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)

    input = input - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True)

    scale = furthest_distance
    input = input / scale
    return torch.from_numpy(input), torch.from_numpy(centroid), torch.from_numpy(scale)

def normal_pc(p1,p2):
    p1, centroid, scale = normalize_point_cloud(p1.cpu().numpy())  # BN3, B13, B11
    #p1 = p1.cuda()
    #scale = scale.cuda()
    p2 = (p2 - centroid) / scale

    return p1, p2, centroid, scale

def denormal_pc(pc, centroid, scale):
    pc = pc * scale + centroid
    return pc


def parse_args():
    parser = argparse.ArgumentParser(description='SRNet')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='cpu')
    parser.add_argument('--test_root', type=str, default='')
    parser.add_argument('--npoints', type=int, default=1024)
    parser.add_argument('--pretrain_flow_net', type=str, default='flow_finetuned/flownet.pth')
    parser.add_argument('--pretrain_model', type=str, default='SRmodel/model_for_8ivslf.pth')
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--k_warping', type=int, default=3)
    parser.add_argument('--down_num', type=int, default=256)
    parser.add_argument('--interframes', type=int, default=4)
    parser.add_argument('--dgk', type=int, default=16)
    parser.add_argument('--fpk', type=int, default=16)
    parser.add_argument('--is_8ivfb', type=bool, default=True)
    parser.add_argument('--use_bn', type=bool, default=False)
    return parser.parse_args()

class EMDloss(nn.Module):
    def __init__(self):
        super().__init__()
        import sys
        sys.path.append('./p1_EMDloss')
        from emd import earth_mover_distance_raw
        global earth_mover_distance_raw

    @staticmethod
    def cal_emd(P1, P2):
        N1, N2 = P1.size(1), P2.size(1)
        dist = earth_mover_distance_raw(P1, P2, transpose=False)
        emd_loss = (dist / N1).mean()
        return emd_loss

    def forward(self , pc_gen, pc_gt):
        loss = EMDloss.cal_emd(pc_gen, pc_gt)
        return loss

def loss_compute(pc,gt):
    len_tuple = pc.size(0)
    pc_tuple = torch.split(pc,1,dim=0)
    gt_tuple = torch.split(gt,1,dim=0)
    loss_tol1 = 0
    loss_tol2 = 0
    chamfer_dis = ChamferDistance()
    for l in range(0, len_tuple):
        dist1, dist2 = chamfer_dis(pc_tuple[l], gt_tuple[l])  # b n 3
        loss_item = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_tol1 = loss_tol1 + loss_item
    emd_distence = EMDloss()
    for l in range(0, len_tuple):
        loss_item = emd_distence(pc_tuple[l], gt_tuple[l])  # b n 3
        loss_tol2 = loss_tol2 + loss_item
    loss1 = loss_tol1 / len_tuple
    loss2 = loss_tol2 / len_tuple
    return loss1, loss2

def test_inter(args):
    os.environ['CUDA_VISIBLE_DEVICE'] = args.gpu
    test_dataset = test_DHBDataset(root=args.test_root, npoints=args.npoints, interframes = args.interframes, is_8ivfb=args.is_8ivfb , train=False)
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False,  ###
                            pin_memory=True,
                            drop_last=False)
    net = SRNet(k_warping = args.k_warping, dgk = args.dgk,fpk = args.fpk, down_num = args.down_num, npoints = args.npoints, freeze = args.freeze, use_bn=args.use_bn).cuda()
    net.eval()
    net.flow.load_state_dict(torch.load(args.pretrain_flow_net))
    net.load_state_dict(torch.load(args.pretrain_model))
    metrics = defaultdict(lambda: list())
    pbar = tqdm(enumerate(test_loader))

    loss1_1 = []
    loss2_1 = []
    loss3_1 = []
    loss1_2 = []
    loss2_2 = []
    loss3_2 = []

    for i, data in pbar:
        with torch.no_grad():
            ini_pc, mid_pc, end_pc, ini_color, mid_color, end_color, t = data
            end_pc_n, ini_pc_n, centroid, scale = normal_pc(end_pc, ini_pc)
            ini_pc_n = ini_pc_n.cuda(non_blocking=True)
            end_pc_n = end_pc_n.cuda(non_blocking=True)
            ini_color = ini_color.cuda(non_blocking=True)
            end_color = end_color.cuda(non_blocking=True)
            centroid = centroid.cuda(non_blocking=True)
            scale = scale.cuda(non_blocking=True)

            inter_loss1 = 0
            inter_loss2 = 0

            for k in range(0, args.interframes):
                mid_color[k] = mid_color[k].cuda(non_blocking=True)
                mid_pc[k] = mid_pc[k].cuda(non_blocking=True)
                t[k] = t[k].cuda(non_blocking=True).float().unsqueeze(dim=-1)
                pred_mid_pc_n = net(ini_pc_n, end_pc_n, ini_color, end_color, t[k])
                pred_mid_pc = denormal_pc(pred_mid_pc_n, centroid, scale)
                temp1, temp2 = loss_compute(pred_mid_pc, mid_pc[k])
                inter_loss1 = temp1 + inter_loss1
                inter_loss2 = temp2 + inter_loss2
            loss1 = inter_loss1 / args.interframes
            loss2 = inter_loss2 / args.interframes
            # print(loss)
        metrics['cd_test_loss'].append(loss1.cpu().data.numpy())
        metrics['emd_test_loss'].append(loss2.cpu().data.numpy())

    mean_loss1 = np.mean(metrics['cd_test_loss'])
    mean_loss2 = np.mean(metrics['emd_test_loss'])

    return mean_loss1, mean_loss2



if __name__ == "__main__":
    args = parse_args()
    mean_loss1, mean_loss2= test_inter(args)
    print("*******CD*********")
    print("mean_loss_cd:", mean_loss1)
    print("*******EMD*********")
    print("mean_loss_emd:", mean_loss2)

