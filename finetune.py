# 在8iVSLF上finetune代码
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
from chamfer_distance import ChamferDistance
from datasets.dhb_set import train_DHBDataset
from model_icme.SRNet import SRNet
from tqdm import tqdm
import argparse
from model_icme.main_utils import *
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs')

def parse_args():
    parser = argparse.ArgumentParser(description='SRNet')

    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=700)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=0.00004)
    parser.add_argument('--step_size_lr', type=int, default=25)
    parser.add_argument('--gamma_lr', type=float, default=0.7)
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--init_bn_momentum', type=float, default=0.5)
    parser.add_argument('--min_bn_momentum', type=float, default=0.01)
    parser.add_argument('--step_size_bn_momentum', type=int, default=100)
    parser.add_argument('--gamma_bn_momentum', type=float, default=0.5)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--train_root', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='model_pretrained/')
    parser.add_argument('--mid_save_dir', type=str, default='model_mid/')
    parser.add_argument('--save_name', type=str, default='model_for_8ivslf')
    parser.add_argument('--model_for_dhb', type=str, default='SRmodel/model_for_mitama.pth')
    parser.add_argument('--npoints', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='DHB')
    parser.add_argument('--pretrain_flow_model', type=str, default='flow_finetuned/flownet.pth')
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--k_warping', type=int, default=3)
    parser.add_argument('--down_num', type=int, default=256)
    parser.add_argument('--interframes', type=int, default=3)
    parser.add_argument('--dgk', type=int, default=16)
    parser.add_argument('--fpk', type=int, default=16)
    parser.add_argument('--use_bn', type=bool, default=False)
    return parser.parse_args()

def loss_compute(pc,gt):
    len_tuple = pc.size(0)
    pc_tuple = torch.split(pc,1,dim=0)
    gt_tuple = torch.split(gt,1,dim=0)
    loss_tol = 0
    chamfer_dis = ChamferDistance()
    for l in range(0, len_tuple):
        dist1, dist2 = chamfer_dis(pc_tuple[l], gt_tuple[l])  # b n 3
        loss_item = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_tol = loss_tol + loss_item
    loss = loss_tol / len_tuple
    return loss

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

def train_inter(args):
    os.environ['CUDA_VISIBLE_DEVICE'] = args.gpu
    train_dataset = train_DHBDataset(root=args.train_root, npoints=args.npoints, interframes = args.interframes, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False)
    net = SRNet(k_warping = args.k_warping, dgk = args.dgk, fpk = args.fpk, down_num = args.down_num, npoints = args.npoints, freeze = args.freeze, use_bn=args.use_bn).cuda()
    net.flow.load_state_dict(torch.load(args.pretrain_flow_model))
    net.load_state_dict(torch.load(args.model_for_dhb))
    optimizer = optim.Adam(net.parameters(), lr=args.init_lr)
    # 设置为根据验证loss自动调整学习率：当验证集的损失不下降时降低0.5倍
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.factor, patience=args.step_size_lr, verbose=True,
                                                           min_lr=args.min_lr)

    def update_bn_momentum(epoch):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = max(
                    args.init_bn_momentum * args.gamma_bn_momentum ** (epoch // args.step_size_bn_momentum),
                    args.min_bn_momentum)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        update_bn_momentum(epoch)
        net.train()
        count = 0
        total_loss = 0
        pbar = tqdm(enumerate(train_loader))

        for i, data in pbar:
            ini_pc, mid_pc, end_pc, ini_color, mid_color, end_color, t = data
            ini_pc_n, end_pc_n, centroid, scale = normal_pc(ini_pc, end_pc)
            ini_pc_n = ini_pc_n.cuda(non_blocking=True)
            end_pc_n = end_pc_n.cuda(non_blocking=True)
            ini_color = ini_color.cuda(non_blocking=True)
            end_color = end_color.cuda(non_blocking=True)
            centroid = centroid.cuda(non_blocking=True)
            scale = scale.cuda(non_blocking=True)

            optimizer.zero_grad()
            inter_loss = 0

            for k in range(0, args.interframes):
                mid_color[k] = mid_color[k].cuda(non_blocking=True)
                mid_pc[k] = mid_pc[k].cuda(non_blocking=True)
                t[k] = t[k].cuda(non_blocking=True).float().unsqueeze(dim=-1)
                pred_mid_pc_n = net(ini_pc_n, end_pc_n, ini_color, end_color, t[k])
                pred_mid_pc = denormal_pc(pred_mid_pc_n, centroid, scale)
                temp = loss_compute(pred_mid_pc, mid_pc[k])
                inter_loss = temp + inter_loss

            loss = inter_loss/args.interframes
            loss.requires_grad_(True)
            loss.backward()

            optimizer.step()

            count += 1
            total_loss += loss.item()
            if i % 10 == 0:  ###
                pbar.set_description('Train Epoch:{}[{}/{}({:.0f}%)] Loss: {:.6f}'.format(
                    epoch + 1, i, len(train_loader), 100. * i / len(train_loader), loss.item()
                ))

        total_loss = total_loss / count
        print('Epoch ', epoch + 1, 'finished ', 'train_loss = ', total_loss)

        scheduler.step(total_loss)
        writer.add_scalar("train_loss", total_loss, epoch+1)

        if epoch % 100 == 0:
            torch.save(net.state_dict(), args.mid_save_dir + 'SRNet_ft_' + str(epoch) + args.save_name + '.pth')

        if total_loss < best_loss:
            torch.save(net.state_dict(), args.save_dir + 'SRNet' + args.save_name + '.pth')
            best_loss = total_loss

        print('Best train loss: {:.6f}'.format(best_loss))

    writer.close()



if __name__ == "__main__":
    args = parse_args()
    train_inter(args)
