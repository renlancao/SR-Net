"""
插帧模型总pipeline
"""
import torch
import torch.nn as nn
import numpy as np
from model_icme.flowmodel import PointConvBidirection
from pointnet2 import pointnet2_utils
from model_icme.pointconv_util import knn_point, FrameWarping, MLP
from model_icme.dgcnn import FeatureWarping
import torch.nn.functional as F
import model_icme.pytorch_utils as pt_utils
from chamfer_distance import ChamferDistance

class SRNet(nn.Module):
    def __init__(self, k_warping, dgk, fpk, down_num, npoints, freeze=1, use_bn=False):
        super(SRNet, self).__init__()
        self.flow = PointConvBidirection()
        #self.flow = torch.nn.DataParallel(self.flow)
        if freeze == 1:
            for p in self.parameters():
                p.requires_grad = False

        self.Fwarping = FeatureWarping(dgk=dgk, fpk = fpk, down_num=down_num, npoints=npoints, embed_dim=256)
        self.warping = FrameWarping(k=k_warping)

        self.pcd_layer = nn.Sequential(
            pt_utils.SharedMLP([512, 256], bn=use_bn),
            pt_utils.SharedMLP([256, 64], bn=use_bn),
            pt_utils.SharedMLP([64, 3], activation=None, bn=False))

        self.mlp_layer = nn.Sequential(
            pt_utils.SharedMLP([256, 64], bn=use_bn),
            pt_utils.SharedMLP([64, 1], activation=None, bn=False))

        self.fusion =  nn.Sequential(
            pt_utils.SharedMLP([512, 256], bn=use_bn),
            pt_utils.SharedMLP([256, 256], activation=None, bn=False))

    def forward(self, points1, points2, color1, color2, t):
        '''
        Input:
            points1: [B,N,3]
            points2: [B,N,3]
            color1: [B,N,3]
            color2: [B,N,3]
        '''

        # Estimate 3D scene flow
        with torch.no_grad():
            flow_forward, _, _, _, _ = self.flow(points1, points2, color1, color2)
            flow_backward, _, _, _, _ = self.flow(points2, points1, color2, color1)

        flow_forward[0] = flow_forward[0].transpose(1, 2).contiguous()
        flow_backward[0] = flow_backward[0].transpose(1, 2).contiguous()
        # B n 3

        # warp
  
        t = t.unsqueeze(1)
        warped_points1_xyz = points1 + flow_forward[0] * t
        warped_points2_xyz = points2 + flow_backward[0] * (1 - t)
        # B N 3

        map1 = self.Fwarping(points1)
        map2 = self.Fwarping(points2)
        # B 256 N
        f1 = self.warping(points1, warped_points1_xyz, map1)
        # warped_points1_xyz与points1 warping
        # f1:B N C

        f2 = self.warping(points2, warped_points2_xyz, map2)
        # f2:B N C
        # sampling

        f = torch.cat((f1, f2), dim=2)
        #B N 2C
        f = f.transpose(1, 2).contiguous()
        # B 2C N
        f = f.unsqueeze(-1)
        # B 2C N 1
        coarse = self.pcd_layer(f).squeeze(-1)
        coarse = coarse.transpose(1, 2).contiguous()
        # B N 3

        wf1 = f1.transpose(1, 2).contiguous()
        wf1 = wf1.unsqueeze(-1)
        wf2 = f2.transpose(1, 2).contiguous()
        wf2 = wf2.unsqueeze(-1)
        # MLP input B C N 1
        w1 = self.mlp_layer(wf1)
        # 过一系列MLP，得到 B 1 N 1
        w1 = w1.squeeze(1)
        # 得到 B N 1
        w2 = self.mlp_layer(wf2)
        w2 = w2.squeeze(1)
        # 得到 B N 1

        # B N 1

        avg1 = torch.mean(w1, dim=1, keepdim=True)
        # 在点数维度进行平均 B 1 1
        avg2 = torch.mean(w2, dim=1, keepdim=True)
        # B 1 1
        w = F.softmax(torch.stack((avg1, avg2), 0), dim=0)
        w1, w2 = w.split(1, 0)
        w1 = w1.squeeze(dim=0)*(1-t)
        w2 = w2.squeeze(dim=0)*t

        gated_f1 = torch.mul(w1, f1)
        gated_f2 = torch.mul(w2, f2)
        # B N C
        out = torch.cat((gated_f1, gated_f2), 2)

        out = out.transpose(1, 2).contiguous()
        out = out.unsqueeze(-1)

        delta = self.pcd_layer(out)
        #new_points = self.pcd_layer(base_f)
        # 通过一系列MLP获得最终的帧 B N 3
        delta = delta.squeeze(-1)

        delta = delta.transpose(1, 2).contiguous()

        new_points = coarse + delta

        return new_points
        # B N 3
