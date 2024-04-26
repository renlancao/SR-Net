import torch
from torch import nn
from pointnet2 import pointnet2_utils
from model_icme.pointconv_util import knn_point, FeaturePropagation

class DGCNN_Grouper(nn.Module):
    def __init__(self, down_num, npoints, dgk):
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 32),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )
        self.down_num = down_num
        self.npoints = npoints
        self.dgk = dgk

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k, dgk):
        # coor: bs, 3, np, x: bs, c, np
        '''
        knn(ref, query)
        返回的是n_q
        '''
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)
        # distances, indices = knn.knn(query_points, reference_points, K)
        # 这里没太看懂
        with torch.no_grad():
            idx = knn_point(dgk, xyz=coor_k.transpose(2,1).contiguous(), new_xyz=coor_q.transpose(2,1).contiguous())  # bs k np # 返回离coor_q中最近的k个点的idx
            idx = idx.transpose(2,1).contiguous()
            assert idx.shape[1] == dgk
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1,
                                                                           1) * num_points_k  # [batch_size, 1, 1]
            idx = idx + idx_base  # 方便维度变化
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        # [idx,:]有点奇怪
        feature = x_k.view(batch_size * num_points_k, -1)[idx,
                  :]  # 离coor_q最近的k个点的特征(batch_size * num_points_q*k, num_dims)
        feature = feature.view(batch_size, dgk, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()  # 维度变化
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, dgk)  # x_q 复制k倍
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x):
        # x: bs, 3, n
        coor = x   
        f = self.input_trans(x)
        inpc_f = f
        f = self.get_graph_feature(coor, f, coor, f, self.dgk)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, self.npoints//2)
        f = self.get_graph_feature(coor_q, f_q, coor, f, self.dgk)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        xyz1, point1 = coor_q, f
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f, self.dgk)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, self.down_num)
        f = self.get_graph_feature(coor_q, f_q, coor, f, self.dgk)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        return coor, f, xyz1, point1, inpc_f


class FeatureWarping(nn.Module):

    def __init__(self, dgk, fpk, down_num, npoints, embed_dim):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # 特征维度
        self.grouper = DGCNN_Grouper(down_num = down_num, npoints = npoints, dgk = dgk)  # B 3 N to B C(3) N(128) and B C(128) N(128)    #DGCNN特征提取器

        self.input_proj = nn.Sequential(
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        self.fp1 = FeaturePropagation(in_channels1 = 256, in_channels2 = 64, out_channels = [256, 256], k = fpk)

        self.fp2 = FeaturePropagation(in_channels1 = 256, in_channels2 = 8, out_channels = [256, 256], k = fpk)

    def forward(self, inpc):
        '''
            inpc : input incomplete point cloud with shape B N C(3)
        '''
        inpc = inpc.transpose(1, 2).contiguous()
        coor, f, xyz1, point1, inpc_f = self.grouper(inpc)  # 得到N个局部区域的特征
        # coor [B,3,256]B C N
        # f [B,128,256] B C N
        # xyz1 [B,3,1024] B C N
        # points1 [B,64,1024]B C N
        # inpc_f [B,8,N]
        # b n c
        x = self.input_proj(f).transpose(1, 2)

        new_features1 = self.fp1(coor.transpose(1, 2).contiguous(), xyz1.transpose(1, 2).contiguous(), x, point1)
        # B C N
        new_features2 = self.fp2(xyz1.transpose(1, 2).contiguous(), inpc.transpose(1, 2).contiguous(),
                                 new_features1.transpose(1, 2).contiguous(), inpc_f)

        return new_features2
        # B C N(B 256 N)
