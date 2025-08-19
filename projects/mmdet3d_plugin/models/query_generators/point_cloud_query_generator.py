# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
import torch_scatter

from ..builder import QUERY_GENERATORS


@QUERY_GENERATORS.register_module()
class PointCloudQueryGenerator(BaseModule):
    def __init__(self, in_channels=128, hidden_channel=128, pts_use_cat=False,
                 dataset='nuscenes', virtual_voxel_size=None, point_cloud_range=None, head_pc_range=None):
        super(PointCloudQueryGenerator, self).__init__()

        assert dataset in ['nuscenes', 'argov2']

        # a shared convolution
        self.empty_pos = nn.Embedding(100000, 3)
        self.empty_embed = nn.Embedding(1, in_channels)

        self.pre_bev_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channel),
            nn.LayerNorm(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel, hidden_channel),
        )
        self.bev_embed = nn.Identity()
        self.query_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channel),
            nn.LayerNorm(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel, hidden_channel),
        )
        self.query_pred_embed = nn.Sequential(
            nn.Linear(7 * 32 if dataset == 'nuscenes' else 5 * 32, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, hidden_channel)
        )

        self.pts_use_cat = pts_use_cat
        if self.pts_use_cat:
            num_cls = 10 if dataset == 'nuscenes' else 26
            self.pts_cat_embed = nn.Embedding(num_cls, hidden_channel)

        self.virtual_voxel_size = virtual_voxel_size
        self.point_cloud_range = point_cloud_range
        self.head_pc_range = head_pc_range

    def init_weights(self):
        super(PointCloudQueryGenerator, self).init_weights()
        nn.init.uniform_(self.empty_pos.weight.data, 0, 1)
        self.empty_pos.weight.requires_grad = False

    @staticmethod
    def pos2embed(pos, num_pos_feats=128, temperature=10000):
        import math
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
        pos_x = pos[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        return pos_x.flatten(-2)

    def forward(self, lidar_feat, lidar_indices, lidar_xyz, query_feat, query_xyz, query_pred, query_cat, batch_size):
        # 断言：激光雷达索引和坐标不需要梯度（非可学习参数）
        assert not lidar_indices.requires_grad and not lidar_xyz.requires_grad
        # 断言：查询相关的坐标、预测、类别和特征均不需要梯度（通常来自检测头输出，非当前模块优化目标）
        assert not any(x.requires_grad for x in query_xyz) and not any(x.requires_grad for x in query_pred) and \
               not any(x.requires_grad for x in query_cat) and not any(x.requires_grad for x in query_feat)

        device = lidar_feat.device
        voxel_size = torch.tensor(self.virtual_voxel_size, device=device)# 体素大小（预设参数）
        pc_range = torch.tensor(self.point_cloud_range, device=device)# 点云有效范围（x_min, y_min, z_min, x_max, y_max, z_max）

        # 步骤1：提取BEV体素索引（去重），保留批次ID和BEV平面坐标（y, x）
        bev_indices, bev_indices_inv = lidar_indices[:, [0, 2, 3]].unique(dim=0,
                                                                          return_inverse=True)  # [n_v, (bid, y, x)]
        # 步骤2：BEV特征嵌入与聚合
        bev_lidar_feat = self.pre_bev_embed(lidar_feat) + lidar_feat
        # 按BEV体素索引聚合特征（同一体素内特征平均）
        bev_feat = torch_scatter.scatter(bev_lidar_feat, bev_indices_inv, dim=0, reduce='mean')
        # 步骤3：计算BEV体素中心坐标（仅x/y平面，忽略z）
        bev_xyz = (bev_indices[:, [2, 1]] + 0.5) * voxel_size[None, :2] + pc_range[None, :2]
        # 更新特征和坐标为BEV格式
        lidar_feat = bev_feat # 现在是BEV支柱特征
        lidar_indices = bev_indices # BEV体素索引
        lidar_xyz = bev_xyz# BEV体素中心坐标（x/y）

        # generate query content features from 3D detection
        #处理查询特征
        # 步骤1：处理空查询（若样本无查询特征，用零矩阵填充）
        query_feat = [x if len(x) > 0 else x.new_zeros((0, 128)) for x in query_feat]
        # 步骤2：查询特征嵌入增强
        query_feat = [self.query_embed(x) + x for x in query_feat]
        # 步骤3：融入预测结果信息（如边界框参数）
        query_feat_w_pred = [x + self.query_pred_embed(self.pos2embed(pred, 32, temperature=20)) for x, pred in
                             zip(query_feat, query_pred)]
        # 步骤4：（可选）融入类别信息
        if self.pts_use_cat:
            query_feat_w_pred = [x + self.pts_cat_embed(cat) for x, cat in zip(query_feat_w_pred, query_cat)]
        # 更新查询特征和坐标
        query_feat = query_feat_w_pred # 最终查询特征（融合原始特征+预测+类别）
        query_xyz = query_xyz# 保持查询坐标不变

        #填充激光雷达特征与位置（统一批次形状）
        
        feat_size = lidar_feat.size(-1)# 特征维度
        # 步骤1：生成批次掩码（区分不同样本的激光雷达特征）
        batch_mask = [lidar_indices[:, 0] == b for b in range(batch_size)]# 每个样本的掩码（True表示属于该样本）
        lidar_size = [m.sum().item() for m in batch_mask] # 每个样本的激光雷达特征数量
        max_size = max(lidar_size)# 批次内最大特征数量（用于填充）

        # pad key/value positions
        # 步骤2：填充激光雷达位置（BEV坐标归一化）
        head_pc_range = torch.tensor(self.head_pc_range, device=device)
        # 将BEV坐标归一化到[0,1]区间（相对于head_pc_range）
        lidar_xyz = (lidar_xyz[:, :2] - head_pc_range[:2]) / (head_pc_range[3:5] - head_pc_range[:2])
        # 初始化位置矩阵（用0.5填充，后续替换有效位置）
        lidar_pos = lidar_feat.new_zeros([batch_size, max_size, 2]) + 0.5
        # 填充空位置（若特征数量不足max_size，用预设的空位置嵌入填充）
        pad_size = min(max_size, self.empty_pos.weight.size(0))
        lidar_pos[:, -pad_size:] = self.empty_pos.weight[:pad_size, :2]# empty_pos为可学习的空位置嵌入
        # 替换每个样本的有效位置
        for b in range(batch_size):
            lidar_pos[b, :lidar_size[b]] = lidar_xyz[batch_mask[b]][..., :2]# 有效位置替换为归一化坐标

        # pad key/value features
        # 步骤3：填充激光雷达特征（统一长度）
        # 初始化特征矩阵（用空特征嵌入填充）
        lidar_feat_in = lidar_feat.new_zeros([batch_size, max_size, feat_size]) + self.empty_embed.weight# empty_embed为可学习的空特征
        # 替换每个样本的有效特征
        for b in range(batch_size):
            lidar_feat_in[b, :lidar_size[b]] = lidar_feat[batch_mask[b]]# 有效特征替换为实际特征
        lidar_feat = lidar_feat_in# 更新为填充后的特征
        # 填充查询特征与位置
        # pad query positions
        # 步骤1：确定查询特征的最大长度
        max_size = max(len(x) for x in query_feat)
        # 步骤2：填充查询位置（3D坐标）
        query_pos = lidar_feat.new_zeros([batch_size, max_size, 3])
        ## 填充空查询位置（用预设空位置嵌入，映射到实际空间范围）
        pad_size = min(max_size, self.empty_pos.weight.size(0))
        query_pos[:, -pad_size:] = self.empty_pos.weight[:pad_size, :3] * (
                head_pc_range[3:6] - head_pc_range[0:3]) + head_pc_range[0:3]# 空位置嵌入→实际坐标
        # 替换每个样本的有效查询位置
        for b in range(batch_size):
            query_pos[b, :len(query_feat[b])] = query_xyz[b][..., :3]
        # 步骤3：填充查询特征（统一长度）
        # 初始化查询特征矩阵（用空特征嵌入填充）
        query_feat_in = lidar_feat.new_zeros([batch_size, max_size, feat_size]) + self.empty_embed.weight
        # 替换每个样本的有效查询特征
        for b in range(batch_size):
            if len(query_feat[b]) > 0:
                query_feat_in[b, :len(query_feat[b])] = query_feat[b] # 有效特征替换为实际特征
        query_feat = query_feat_in# 更新为填充后的查询特征
        #返回填充后的激光雷达特征、激光雷达位置、查询特征和查询位置
        return lidar_feat, lidar_pos, query_feat, query_pos
    
    # forward方法的核心是将激光雷达特征从 3D 体素转换为 2D BEV 特征，并增强检测查询特征的语义信息，最终通过填充统一批次内的特征形状。具体流程可概括为：

    # 3D→BEV 转换：通过体素聚合将 3D 激光雷达特征投影到 BEV 平面，减少维度并聚焦平面空间关系；
    # 查询特征增强：融合查询特征与预测参数、类别标签，提升语义表达；
    # 统一形状填充：通过可学习的空嵌入填充特征和位置，确保批次内所有样本的特征形状一致，为后续网络（如 Transformer）提供规范输入。
