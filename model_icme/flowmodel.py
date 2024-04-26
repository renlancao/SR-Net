"""
PointConvBidirection
用于插帧的光流模型
"""

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from model_icme.pointconv_util import PointConvD, PointWarping, UpsampleFlow, CrossLayerLight as CrossLayer
from model_icme.pointconv_util import SceneFlowEstimatorResidual
from model_icme.pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance
import time

scale = 1.0

class PointConvBidirection(nn.Module):
    def __init__(self):
        super(PointConvBidirection, self).__init__()

        flow_nei = 32
        feat_nei = 16
        self.scale = scale
        #l0: 8192
        self.level0 = Conv1d(3, 32)
        self.level0_1 = Conv1d(32, 32)
        # self.level0_1_t1 = Conv1d(32, 32)
        # self.level0_1_t2 = Conv1d(32, 32)
        self.cross0 = CrossLayer(flow_nei, 32 + 32 , [32, 32], [32, 32])
        self.flow0 = SceneFlowEstimatorResidual(32 + 64, 32)
        self.level0_2 = Conv1d(32, 64)

        #l1: 2048
        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64)
        self.cross1 = CrossLayer(flow_nei, 64 + 32, [64, 64], [64, 64])
        self.flow1 = SceneFlowEstimatorResidual(64 + 64, 64)
        self.level1_0 = Conv1d(64, 64)
        # self.level1_0_t1 = Conv1d(64, 64)
        # self.level1_0_t2 = Conv1d(64, 64)
        self.level1_1 = Conv1d(64, 128)

        #l2: 512
        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128)
        self.cross2 = CrossLayer(flow_nei, 128 + 64, [128, 128], [128, 128])
        self.flow2 = SceneFlowEstimatorResidual(128 + 64, 128)
        self.level2_0 = Conv1d(128, 128)
        # self.level2_0_t1 = Conv1d(128, 128)
        # self.level2_0_t2 = Conv1d(128, 128)
        self.level2_1 = Conv1d(128, 256)

        #l3: 256
        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256)
        self.cross3 = CrossLayer(flow_nei, 256 + 64, [256, 256], [256, 256])
        self.flow3 = SceneFlowEstimatorResidual(256, 256)
        self.level3_0 = Conv1d(256, 256)
        # self.level3_0_t1 = Conv1d(256, 256)
        # self.level3_0_t2 = Conv1d(256, 256)
        self.level3_1 = Conv1d(256, 512)

        #l4: 64
        self.level4 = PointConvD(64, feat_nei, 512 + 3, 256)

        #deconv
        # self.deconv4_3_t1 = Conv1d(256, 64)
        # self.deconv4_3_t2 = Conv1d(256, 64)
        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 64)
        self.deconv2_1 = Conv1d(128, 32)
        self.deconv1_0 = Conv1d(64, 32)

        #warping
        self.warping = PointWarping()

        #upsample
        self.upsample = UpsampleFlow()

    def forward(self, xyz1, xyz2, color1, color2):
       
        #xyz1, xyz2: B, N, 3
        #color1, color2: B, N, 3

        #l0
        pc1_l0 = xyz1.permute(0, 2, 1)
        pc2_l0 = xyz2.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1) # B 3 N
        color2 = color2.permute(0, 2, 1) # B 3 N
        feat1_l0 = self.level0(color1)
        # feat1_l0 = self.level0_1_t1(feat1_l0)
        feat1_l0 = self.level0_1(feat1_l0)
        feat1_l0_1 = self.level0_2(feat1_l0)

        feat2_l0 = self.level0(color2)
        # feat2_l0 = self.level0_1_t2(feat2_l0)
        feat2_l0 = self.level0_1(feat2_l0)
        feat2_l0_1 = self.level0_2(feat2_l0)

        #l1
        pc1_l1, feat1_l1, fps_pc1_l1 = self.level1(pc1_l0, feat1_l0_1)
        # feat1_l1 = self.level1_0_t1(feat1_l1)
        feat1_l1 = self.level1_0(feat1_l1)
        feat1_l1_2 = self.level1_1(feat1_l1)

        pc2_l1, feat2_l1, fps_pc2_l1 = self.level1(pc2_l0, feat2_l0_1)
        # feat2_l1 = self.level1_0_t2(feat2_l1)
        feat2_l1 = self.level1_0(feat2_l1)
        feat2_l1_2 = self.level1_1(feat2_l1)

        #l2
        pc1_l2, feat1_l2, fps_pc1_l2 = self.level2(pc1_l1, feat1_l1_2)
        # feat1_l2 = self.level2_0_t1(feat1_l2)
        feat1_l2 = self.level2_0(feat1_l2)
        feat1_l2_3 = self.level2_1(feat1_l2)

        pc2_l2, feat2_l2, fps_pc2_l2 = self.level2(pc2_l1, feat2_l1_2)
        # feat2_l2 = self.level2_0_t2(feat2_l2)
        feat2_l2 = self.level2_0(feat2_l2)
        feat2_l2_3 = self.level2_1(feat2_l2)

        #l3
        pc1_l3, feat1_l3, fps_pc1_l3 = self.level3(pc1_l2, feat1_l2_3)
        # feat1_l3 = self.level3_0_t1(feat1_l3)
        feat1_l3 = self.level3_0(feat1_l3)
        feat1_l3_4 = self.level3_1(feat1_l3)

        pc2_l3, feat2_l3, fps_pc2_l3 = self.level3(pc2_l2, feat2_l2_3)
        # feat2_l3 = self.level3_0_t2(feat2_l3)
        feat2_l3 = self.level3_0(feat2_l3)
        feat2_l3_4 = self.level3_1(feat2_l3)

        #l4
        pc1_l4, feat1_l4, _ = self.level4(pc1_l3, feat1_l3_4)
        feat1_l4_3 = self.upsample(pc1_l3, pc1_l4, feat1_l4)
        # feat1_l4_3 = self.deconv4_3_t1(feat1_l4_3)
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)

        pc2_l4, feat2_l4, _ = self.level4(pc2_l3, feat2_l3_4)
        feat2_l4_3 = self.upsample(pc2_l3, pc2_l4, feat2_l4)
        # feat2_l4_3 = self.deconv4_3_t2(feat2_l4_3)
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)

        #l3
        c_feat1_l3 = torch.cat([feat1_l3, feat1_l4_3], dim = 1)
        c_feat2_l3 = torch.cat([feat2_l3, feat2_l4_3], dim = 1)
        feat1_new_l3, feat2_new_l3, cross3 = self.cross3(pc1_l3, pc2_l3, c_feat1_l3, c_feat2_l3)
        feat3, flow3 = self.flow3(pc1_l3, feat1_l3, cross3)

        feat1_l3_2 = self.upsample(pc1_l2, pc1_l3, feat1_new_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)

        feat2_l3_2 = self.upsample(pc2_l2, pc2_l3, feat2_new_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)

        c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim = 1)
        c_feat2_l2 = torch.cat([feat2_l2, feat2_l3_2], dim = 1)

        #l2
        up_flow2 = self.upsample(pc1_l2, pc1_l3, self.scale * flow3)
        pc2_l2_warp = self.warping(pc1_l2, pc2_l2, up_flow2)
        feat1_new_l2, feat2_new_l2, cross2 = self.cross2(pc1_l2, pc2_l2_warp, c_feat1_l2, c_feat2_l2)

        feat3_up = self.upsample(pc1_l2, pc1_l3, feat3)
        new_feat1_l2 = torch.cat([feat1_l2, feat3_up], dim = 1)
        feat2, flow2 = self.flow2(pc1_l2, new_feat1_l2, cross2, up_flow2)

        feat1_l2_1 = self.upsample(pc1_l1, pc1_l2, feat1_new_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)

        feat2_l2_1 = self.upsample(pc2_l1, pc2_l2, feat2_new_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)

        c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim = 1)
        c_feat2_l1 = torch.cat([feat2_l1, feat2_l2_1], dim = 1)

        #l1
        up_flow1 = self.upsample(pc1_l1, pc1_l2, self.scale * flow2)
        pc2_l1_warp = self.warping(pc1_l1, pc2_l1, up_flow1)
        feat1_new_l1, feat2_new_l1, cross1 = self.cross1(pc1_l1, pc2_l1_warp, c_feat1_l1, c_feat2_l1)

        feat2_up = self.upsample(pc1_l1, pc1_l2, feat2)
        new_feat1_l1 = torch.cat([feat1_l1, feat2_up], dim = 1)
        feat1, flow1 = self.flow1(pc1_l1, new_feat1_l1, cross1, up_flow1)

        feat1_l1_0 = self.upsample(pc1_l0, pc1_l1, feat1_new_l1)
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)

        feat2_l1_0 = self.upsample(pc2_l0, pc2_l1, feat2_new_l1)
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)

        c_feat1_l0 = torch.cat([feat1_l0, feat1_l1_0], dim = 1)
        c_feat2_l0 = torch.cat([feat2_l0, feat2_l1_0], dim = 1)

        #l0
        up_flow0 = self.upsample(pc1_l0, pc1_l1, self.scale * flow1)
        pc2_l0_warp = self.warping(pc1_l0, pc2_l0, up_flow0)
        _, _, cross0 = self.cross0(pc1_l0, pc2_l0_warp, c_feat1_l0, c_feat2_l0)

        feat1_up = self.upsample(pc1_l0, pc1_l1, feat1)
        new_feat1_l0 = torch.cat([feat1_l0, feat1_up], dim = 1)
        _, flow0 = self.flow0(pc1_l0, new_feat1_l0, cross0, up_flow0)

        flows = [flow0, flow1, flow2, flow3]
        pc1 = [pc1_l0, pc1_l1, pc1_l2, pc1_l3]
        pc2 = [pc2_l0, pc2_l1, pc2_l2, pc2_l3]
        fps_pc1_idxs = [fps_pc1_l1, fps_pc1_l2, fps_pc1_l3]
        fps_pc2_idxs = [fps_pc2_l1, fps_pc2_l2, fps_pc2_l3]

        return flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2
