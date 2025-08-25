import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result

from mmdet.models import build_detector, build_backbone, build_head, build_neck
from mmdet3d.models import builder

from mmdet3d.ops import Voxelization, furthest_point_sample
from .ops.sst.sst_ops import scatter_v2, get_inner_win_inds, build_mlp
from scipy.sparse.csgraph import connected_components
from mmdet.core import multi_apply
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
from mmdet3d.models.segmentors.base import Base3DSegmentor

from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential


def filter_almost_empty(coors, min_points):
    new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
    cnt_per_point = unq_cnt[unq_inv]
    valid_mask = cnt_per_point >= min_points
    return valid_mask


@DETECTORS.register_module()
class SingleStageFSDV2(SingleStage3DDetector):

    def __init__(self,
                 backbone,
                 segmentor,
                 voxel_layer=None,
                 voxel_encoder=None,
                 middle_encoder=None,
                 neck=None,
                 virtual_point_projector=None,
                 pre_voxel_encoder=None,
                 bbox_head=None,
                 norm_eval=False,
                 freeze=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 multiscale_cfg=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)

        self.runtime_info = dict()

        if voxel_layer is not None:
            self.voxel_layer = Voxelization(**voxel_layer)

        if voxel_encoder is not None:
            self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
            self.virtual_voxel_size = voxel_encoder['voxel_size']
            self.point_cloud_range = voxel_encoder['point_cloud_range']

        if middle_encoder is not None:
            self.middle_encoder = builder.build_middle_encoder(middle_encoder)

        self.segmentor = build_detector(segmentor)
        self.use_multiscale_features = self.segmentor.use_multiscale_features
        self.head_type = bbox_head['type']
        self.num_classes = bbox_head['num_classes']

        self.cfg = self.train_cfg if self.train_cfg else self.test_cfg
        self.print_info = {}
        self.as_rpn = bbox_head.get('as_rpn', False)

        vpp = virtual_point_projector
        self.virtual_proj = build_mlp(vpp['in_channels'], vpp['hidden_dims'], vpp['norm_cfg'])
        self.ori_proj = build_mlp(vpp['ori_in_channels'], vpp['ori_hidden_dims'], vpp['norm_cfg'])
        self.zero_virtual_feature = vpp.get('zero_virtual_feature', False)
        self.only_virtual = vpp.get('only_virtual', False)

        if self.as_rpn:
            self.recover_proj = build_mlp(vpp['recover_in_channels'], vpp['recover_hidden_dims'], vpp['norm_cfg'])

        self.baseline_mode = self.cfg.get('baseline_mode', False)
        if self.baseline_mode:
            self.virtual_proj = None
            self.ori_proj = None

        self.multiscale_cfg = multiscale_cfg
        if multiscale_cfg is not None:
            ms_projs = []
            for proj in multiscale_cfg['projector_hiddens']:
                this_projector = build_mlp(proj[0], proj[1:], multiscale_cfg['norm_cfg'])
                ms_projs.append(this_projector)
            self.ms_projectors = torch.nn.ModuleList(ms_projs)

        self.norm_eval = norm_eval
        self.freeze = freeze

    def init_weights(self) -> None:
        super(SingleStageFSDV2, self).init_weights()
        if self.freeze:
            for p in self.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        from torch.nn.modules.batchnorm import _BatchNorm
        super(SingleStageFSDV2, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    @torch.no_grad()
    @force_fp32()
    def voxelize_with_batch_idx(self, points, batch_idx):
        """Apply dynamic voxelization to points.
        """
        points = points[:, :3]
        device = points.device
        voxel_size = torch.tensor(self.virtual_voxel_size, device=device)
        pc_range = torch.tensor(self.point_cloud_range, device=device)
        # 计算体素坐标（xyz 到体素索引的转换）
        res_coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        res_coors = res_coors[:, [2, 1, 0]] # to zyx order

        coors_batch = torch.cat([batch_idx[:, None], res_coors], dim=1)

        return coors_batch#【N, 4】

    def clip_points(self, points, pc_range):
        eps = 1e-5
        points[:, 0] = points[:, 0].clamp(min=pc_range[0] + eps, max=pc_range[3] - eps)
        points[:, 1] = points[:, 1].clamp(min=pc_range[1] + eps, max=pc_range[4] - eps)
        points[:, 2] = points[:, 2].clamp(min=pc_range[2] + eps, max=pc_range[5] - eps)
        return points

    def recover_point_features(self, out_voxel_feats, out_coors, out_sparse_shape, cat_pts, cat_batch_idx, voxel_encoder_coors, voxel_encoder_inv):

        is_same = (out_coors == voxel_encoder_coors).all()
        device = out_voxel_feats.device

        if is_same:
            # Meaning that spconv does not change the spatial layout of voxels
            voxel_size = torch.tensor(self.virtual_voxel_size, device=device) # correct only if is_same 
            pc_range = torch.tensor(self.point_cloud_range, device=device)

            voxel_coors_per_pts = out_coors[voxel_encoder_inv]
            feat_per_pts = out_voxel_feats[voxel_encoder_inv]
            voxel_center_per_pts = (voxel_coors_per_pts[:, [3, 2, 1]] + 0.5) * voxel_size[None, :] + pc_range[None, :3]
            offset_per_pts = voxel_center_per_pts - cat_pts
            assert (offset_per_pts.abs() < voxel_size[None, :] / 2 + 1e5).all()

            offset_per_pts = offset_per_pts / voxel_size[None, :] * 2 # normalize

            proj_input = torch.cat([feat_per_pts, offset_per_pts], 1)
            out = self.recover_proj(proj_input)

            return out

        else:
            raise NotImplementedError
    # 主要功能是融合原始点云特征与通过采样得到的 “虚拟点”（预测的目标中心）特征，
    # 经过体素化、编码和 backbone 处理后，提取用于后续检测任务的有效特征
    def extract_feat(self, sampled_dict, origin_dict, gt_bboxes_3d=None, multiscale_features=None):
        """Extract features from points."""
        if self.baseline_mode:
            return self.extract_feat_baseline(sampled_dict, origin_dict, gt_bboxes_3d, multiscale_features)

        sampled_pts = sampled_dict['seg_points']# 采样点的坐标[3000, 5]
        sampled_centers = sampled_dict['center_preds']# 预测的目标中心（虚拟点）[3000, 3]
        sampled_logits = sampled_dict['seg_logits']# 采样点的分割logits[3000, 11]
        sampled_feats = sampled_dict['seg_feats']# 采样点的特征[3000, 131]
        sampled_batch_idx = sampled_dict['batch_idx']# 采样点所属的batch索引[3000]
        device = sampled_pts.device

        # 裁剪预测中心到点云有效范围内（防止超出边界）
        sampled_centers = self.clip_points(sampled_centers, self.point_cloud_range)#[3000, 3]
        # sampled_centers
        # 计算虚拟点与采样点的偏移量（归一化）
        offset = (sampled_centers - sampled_pts[:, :3]) / 10 # hardcode a normalizer[3000,3]
        # 拼接特征：采样点特征 + 偏移量 + 分割logits + 采样点的额外特征（如强度，第3列及以后）
        proj_input = torch.cat([sampled_feats, offset, sampled_logits, sampled_pts[:, 3:]], 1)#[3000, 147]
        vir_pts_feat = self.virtual_proj(proj_input)# 通过虚拟投影层（virtual_proj）生成虚拟点特征[3000, 64]

        if self.zero_virtual_feature:# 若启用，则将虚拟点特征置零（用于消融实验，验证虚拟点的作用）
            vir_pts_feat = vir_pts_feat * 0
        # 从原始字典中提取关键数据
        ori_pts = origin_dict['seg_points']# 原始点坐标及特征[N, 5]
        ori_batch_idx = origin_dict['batch_idx']# 原始点所属的batch索引[N]
        ori_pts_feat = origin_dict['seg_feats'] # 原始点特征[N,131]
        # 通过原始投影层（ori_proj）处理原始点特征（维度调整）
        ori_pts_feat = self.ori_proj(ori_pts_feat)#[N,64]
        # 合并坐标：原始点的x/y/z + 虚拟点（预测中心）的x/y/z
        cat_pts = torch.cat([ori_pts[:, :3], sampled_centers], 0)#[N+3000,3]
        # 合并特征：原始点特征 + 虚拟点特征
        cat_feat = torch.cat([ori_pts_feat, vir_pts_feat], 0)#[N+3000,64]
        # 合并batch索引：原始点的batch索引 + 虚拟点的batch索引
        cat_batch_idx = torch.cat([ori_batch_idx, sampled_batch_idx], 0)#[N+3000]
        # 对合并后的点进行体素化，得到每个点所属的体素坐标（coors）
        coors = self.voxelize_with_batch_idx(cat_pts, cat_batch_idx)#[N+3000, 4]
        # 拼接坐标和特征，作为体素编码器的输入
        voxel_encoder_input = torch.cat([cat_pts, cat_feat], 1)#[N+3000, 67]
        # 体素编码：将同一体素内的点特征聚合为体素特征
        #voxel_feats[m,128] voxel_coors[m,4] 
        voxel_feats, voxel_coors, unq_inv = self.voxel_encoder(voxel_encoder_input, coors, return_inv=True)
        # 生成点类型指示符：0表示原始点，1表示虚拟点
        pts_indicators = torch.cat(
            [
                torch.zeros(len(ori_pts), device=device, dtype=torch.float),
                torch.ones( len(sampled_centers), device=device, dtype=torch.float)
            ]
        )
        # 聚合体素内的指示符（平均），得到体素是否包含虚拟点的标记
        voxel_indicators, scatter_coors = scatter_v2(pts_indicators, coors, mode='avg', return_inv=False)
        assert (scatter_coors == voxel_coors).all()# 确保聚合后的坐标与体素坐标一致
        # 虚拟体素掩码：指示符>0的体素（至少包含一个虚拟点）
        virtual_mask = voxel_indicators > 0

        batch_size = voxel_coors[:, 0].max().item() + 1 # batch大小
        # 若提供多尺度特征，则进行融合（提升特征表达能力）
        if batch_size is not None:
            #[M,128] [M,4] 
            voxel_feats, voxel_coors, singlescale_mask = self.multiscale_fusion(multiscale_features, voxel_feats, voxel_coors)

        if self.only_virtual:# 若仅使用虚拟体素，则过滤掉不含虚拟点的体素
            assert multiscale_features is None
            voxel_feats = voxel_feats[virtual_mask]
            voxel_coors = voxel_coors[virtual_mask]
        # 通过backbone网络（如3D CNN）处理体素特征，得到高层特征
         #[M,128] [M,4] 
        out_voxel_feats, out_coors, sparse_shape = self.backbone(voxel_feats, voxel_coors, batch_size)
        # 若使用多尺度特征，根据掩码筛选原始特征
        if multiscale_features is not None:
            out_voxel_feats = out_voxel_feats[singlescale_mask]#[m, 128]
            out_coors = out_coors[singlescale_mask]#[m, 4]
            voxel_coors = voxel_coors[singlescale_mask] # in fact, out_coors and voxel_coors are same # 确保坐标一致

        # get voxel center xyz# 体素大小和点云范围（预设参数）
        voxel_size = torch.tensor(self.virtual_voxel_size, device=device) # correct only if is_same 
        pc_range = torch.tensor(self.point_cloud_range, device=device)
        # 计算每个体素的中心坐标：体素坐标→实际3D坐标（加0.5*体素大小表示中心）
        voxel_centers = (out_coors[:, [3, 2, 1]] + 0.5) * voxel_size[None, :] + pc_range[None, :3]
        # 输出体素特征、坐标、中心
        out_voxel_dict = {
            'out_voxel_feats': out_voxel_feats,
            'out_coors': out_coors,
            'out_centers': voxel_centers,
        }
        # 提取虚拟体素的特征、坐标、中心（根据是否仅使用虚拟体素）
        if self.only_virtual:
            virtual_voxel_feats = out_voxel_feats
            virtual_coors = out_coors
            virtual_centers = voxel_centers
        else:
            virtual_voxel_feats = out_voxel_feats[virtual_mask]#[Vn,128]
            virtual_coors = out_coors[virtual_mask]#[Vn,4]
            virtual_centers = voxel_centers[virtual_mask]#[Vn,3]
        # 基础输出字典：包含虚拟体素特征及相关信息
        out_dict = dict(    
            virtual_feats=virtual_voxel_feats,
            virtual_coors=virtual_coors,
            virtual_centers=virtual_centers,
            # voxel_indicators=voxel_indicators
            sparse_shape=sparse_shape,# 体素网格的稀疏形状
        )
        out_dict.update(out_voxel_dict)# 合并体素特征信息

        if self.training:
            # 记录虚拟体素数量（用于监控）
            self.print_info['num_virtual'] = virtual_voxel_feats.new_ones(1) * len(virtual_voxel_feats)
            # 计算体素质心（结合真实框的加权质心或平均质心）
            alpha = self.train_cfg.get('centroid_alpha', None)
            if alpha is not None:
                # 计算真实框内的点掩码（用于加权）
                gt_fg_mask = self.get_batched_gt_fg_mask(cat_pts[:, :3], cat_batch_idx, gt_bboxes_3d)
                # 生成权重：真实前景点权重为1，背景点权重为alpha（平衡权重）
                alpha_mask = (~gt_fg_mask).float() * alpha + gt_fg_mask.float()
                 # 加权求和计算体素质心（坐标×权重后求和，再除以权重和）
                sum_centroid, _ = scatter_v2(alpha_mask[:, None] * cat_pts[:, :3], coors, mode='sum', return_inv=False)
                sum_alpha, _ = scatter_v2(alpha_mask[:, None], coors, mode='sum', return_inv=False)
                assert (sum_alpha >= alpha).all()# 确保权重和有效
                voxel_centroid = sum_centroid / sum_alpha
            else:
                 # 简单平均计算体素质心
                voxel_centroid, _ = scatter_v2(cat_pts[:, :3], coors, mode='avg', return_inv=False)
            # 仅保留虚拟体素的质心
            voxel_centroid = voxel_centroid[virtual_mask]
            out_dict['virtual_centroid'] = voxel_centroid
            # 断言：质心必须在体素范围内（误差允许范围内）
            assert ((voxel_centroid - virtual_centers).abs() < voxel_size / 2 + 1e-3).all()

        if self.as_rpn:
            # need pts information for GroupCorrection# 从体素特征恢复点特征（供GroupCorrection等模块使用）
            out_pts_feats = self.recover_point_features(out_voxel_feats, out_coors, sparse_shape, cat_pts, cat_batch_idx, voxel_coors, unq_inv)
            # 补充点相关信息到输出字典
            out_dict['pts_feats'] = out_pts_feats# 恢复的点特征
            out_dict['pts_xyz'] = cat_pts# 合并后的点坐标
            out_dict['pts_indicators'] = pts_indicators# 点类型指示符（原始/虚拟）
            out_dict['pts_batch_inds'] = cat_batch_idx# 点的batch索引

        return out_dict

    def extract_feat_baseline(self, sampled_dict, origin_dict, gt_bboxes_3d=None, multiscale_features=None):
        """Extract features from points."""
        sampled_pts = sampled_dict['seg_points']
        # sampled_centers = sampled_dict['center_preds']
        sampled_centers = sampled_pts[:, :3]
        sampled_logits = sampled_dict['seg_logits']
        sampled_feats = sampled_dict['seg_feats']
        sampled_batch_idx = sampled_dict['batch_idx']
        device = sampled_pts.device

        # the predicted centers might be out-of-range
        # sampled_centers = self.clip_points(sampled_centers, self.point_cloud_range)
        # sampled_centers

        # offset = (sampled_centers - sampled_pts[:, :3]) / 10 # hardcode a normalizer
        # proj_input = torch.cat([sampled_feats, offset, sampled_logits, sampled_pts[:, 3:]], 1)
        # vir_pts_feat = self.virtual_proj(proj_input)

        vir_pts_feat = sampled_feats

        ori_pts = origin_dict['seg_points']
        ori_batch_idx = origin_dict['batch_idx']
        ori_pts_feat = origin_dict['seg_feats']
        # ori_pts_feat = self.ori_proj(ori_pts_feat)

        cat_pts = torch.cat([ori_pts[:, :3], sampled_centers], 0)
        cat_feat = torch.cat([ori_pts_feat, vir_pts_feat], 0)
        cat_batch_idx = torch.cat([ori_batch_idx, sampled_batch_idx], 0)

        coors = self.voxelize_with_batch_idx(cat_pts, cat_batch_idx)

        voxel_encoder_input = torch.cat([cat_pts, cat_feat], 1)
        voxel_feats, voxel_coors, unq_inv = self.voxel_encoder(voxel_encoder_input, coors, return_inv=True)

        pts_indicators = torch.cat(
            [
                torch.zeros(len(ori_pts), device=device, dtype=torch.float),
                torch.ones( len(sampled_centers), device=device, dtype=torch.float)
            ]
        )
        voxel_indicators, scatter_coors = scatter_v2(pts_indicators, coors, mode='avg', return_inv=False)
        assert (scatter_coors == voxel_coors).all()

        batch_size = voxel_coors[:, 0].max().item() + 1
        # assert batch_size == 2, 'Develop assertion, ok to delete'

        if multiscale_features is not None:
            voxel_feats, voxel_coors, singlescale_mask = self.multiscale_fusion(multiscale_features, voxel_feats, voxel_coors)

        out_voxel_feats, out_coors, sparse_shape = self.backbone(voxel_feats, voxel_coors, batch_size)

        if multiscale_features is not None:
            out_voxel_feats = out_voxel_feats[singlescale_mask]
            out_coors = out_coors[singlescale_mask]
            voxel_coors = voxel_coors[singlescale_mask] # in fact, out_coors and voxel_coors are same

        # get voxel center xyz
        voxel_size = torch.tensor(self.virtual_voxel_size, device=device) # correct only if is_same 
        pc_range = torch.tensor(self.point_cloud_range, device=device)
        voxel_centers = (out_coors[:, [3, 2, 1]] + 0.5) * voxel_size[None, :] + pc_range[None, :3]

        virtual_mask = voxel_indicators > 0
        virtual_voxel_feats = out_voxel_feats[virtual_mask]
        virtual_coors = out_coors[virtual_mask]
        virtual_centers = voxel_centers[virtual_mask]

        out_dict = dict(
            virtual_feats=virtual_voxel_feats,
            virtual_coors=virtual_coors,
            virtual_centers=virtual_centers,
            # voxel_indicators=voxel_indicators
            sparse_shape=sparse_shape,
        )

        if self.training:
            self.print_info['num_virtual'] = virtual_voxel_feats.new_ones(1) * len(virtual_voxel_feats)

            alpha = self.train_cfg.get('centroid_alpha', None)
            if alpha is not None:
                gt_fg_mask = self.get_batched_gt_fg_mask(cat_pts[:, :3], cat_batch_idx, gt_bboxes_3d)
                alpha_mask = (~gt_fg_mask).float() * alpha + gt_fg_mask.float()
                sum_centroid, _ = scatter_v2(alpha_mask[:, None] * cat_pts[:, :3], coors, mode='sum', return_inv=False)
                sum_alpha, _ = scatter_v2(alpha_mask[:, None], coors, mode='sum', return_inv=False)
                assert (sum_alpha >= alpha).all()
                voxel_centroid = sum_centroid / sum_alpha
            else:
                voxel_centroid, _ = scatter_v2(cat_pts[:, :3], coors, mode='avg', return_inv=False)

            voxel_centroid = voxel_centroid[virtual_mask]
            out_dict['virtual_centroid'] = voxel_centroid
            assert ((voxel_centroid - virtual_centers).abs() < voxel_size / 2 + 1e-3).all()

        if self.as_rpn:
            # need pts information for GroupCorrection
            out_pts_feats = self.recover_point_features(out_voxel_feats, out_coors, sparse_shape, cat_pts, cat_batch_idx, voxel_coors, unq_inv)
            out_dict['pts_feats'] = out_pts_feats
            out_dict['pts_xyz'] = cat_pts
            out_dict['pts_indicators'] = pts_indicators
            out_dict['pts_batch_inds'] = cat_batch_idx

        return out_dict

    def multiscale_fusion(self, ms_data, voxel_feats, coors):

        cfg = self.multiscale_cfg
        # 从多尺度数据中筛选配置指定的层级（0，1，2）
        ms_data = [ms_data[l] for l in cfg['multiscale_levels']]
        # 对每个尺度的特征进行投影，转换为可融合的维度
        ms_feats = [ self.ms_projectors[i](ms_data[i].features) for i in range(len(ms_data)) ]
        # 对每个尺度的坐标进行投影，转换为与原始体素坐标一致的格式
        ms_coors = [ self.ms_coors_proj(data.indices, data.spatial_shape) for data in ms_data]
        # 计算多尺度特征的总数量（用于生成指示符）
        num_add_feats = sum([len(f) for f in ms_feats])
        # 拼接特征：原始体素特征 + 所有多尺度特征（按第0维拼接）
        cat_feats = torch.cat([voxel_feats,] + ms_feats, 0)
        # 拼接坐标：原始体素坐标 + 所有多尺度特征的坐标（确保特征与坐标一一对应）
        cat_coors = torch.cat([coors,] + ms_coors, 0)
        # 生成指示符：原始体素特征标记为1，多尺度新增特征标记为0
        indicators = torch.cat([voxel_feats.new_ones(len(voxel_feats), 1), voxel_feats.new_zeros(num_add_feats, 1)], 0)
        # 按坐标聚合所有特征（原始+多尺度），融合模式由配置指定（如平均、最大）
        out_feats, out_coors = scatter_v2(cat_feats, cat_coors, mode=cfg['fusion_mode'], return_inv=False)
        # 按坐标聚合指示符（取最大值），标记每个坐标是否包含原始体素特征
        out_indicators, _ = scatter_v2(indicators, cat_coors, mode='max', return_inv=False)
        out_indicators = out_indicators.squeeze()# 去除多余维度（形状从[N,1]变为[N]）
        # 掩码：只保留包含原始体素特征的坐标（指示符为1）
        singlescale_mask = out_indicators == 1
        assert singlescale_mask.sum() == len(voxel_feats)# 断言：原始体素数量与掩码保留的数量一致，确保无丢失
        #融合后的特征（原始体素特征 + 多尺度特征按坐标聚合的结果）
        #融合后特征对应的坐标
        #标记融合特征中属于原始体素的掩码
        return out_feats, out_coors, singlescale_mask

    def ms_coors_proj(self, coors, sparse_shape):

        # support float coors, need inference test
        # cfg = self.multiscale_cfg
        # tgt_sp = cfg['target_sparse_shape']
        # bev_stride = tgt_sp[1] / sparse_shape[1]
        # assert bev_stride == tgt_sp[2] / sparse_shape[2]
        # z_stride = tgt_sp[0] / sparse_shape[0]

        # out_coors = coors.clone()
        # out_coors[:, 1] = (coors[:, 1] * z_stride + z_stride / 2).int()
        # out_coors[:, 2] = (coors[:, 2] * bev_stride + bev_stride / 2).int()
        # out_coors[:, 3] = (coors[:, 3] * bev_stride + bev_stride / 2).int()

        # assert out_coors[:, 1].max().item() < tgt_sp[0]
        # assert out_coors[:, 2].max().item() < tgt_sp[1]
        # assert out_coors[:, 3].max().item() < tgt_sp[2]

        cfg = self.multiscale_cfg
        tgt_sp = cfg['target_sparse_shape']
        bev_stride = tgt_sp[1] // sparse_shape[1]
        assert bev_stride == tgt_sp[2] / sparse_shape[2]
        z_stride = tgt_sp[0] // sparse_shape[0]

        assert z_stride >= 1
        assert bev_stride >= 1

        out_coors = coors.clone()
        out_coors[:, 1] = coors[:, 1] * z_stride + z_stride // 2
        out_coors[:, 2] = coors[:, 2] * bev_stride + bev_stride // 2
        out_coors[:, 3] = coors[:, 3] * bev_stride + bev_stride // 2

        assert out_coors[:, 1].max().item() < tgt_sp[0]
        assert out_coors[:, 2].max().item() < tgt_sp[1]
        assert out_coors[:, 3].max().item() < tgt_sp[2]

        return out_coors

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None,
                      runtime_info=None):
        if runtime_info is not None:
            self.runtime_info = runtime_info # stupid way to get arguements from children class
        # import pdb
        # pdb.set_trace()
        losses = {}
        # 过滤掉标签无效的真值框和标签（仅保留 l >= 0 的标注）
        gt_bboxes_3d = [b[l>=0] for b, l in zip(gt_bboxes_3d, gt_labels_3d)]
        gt_labels_3d = [l[l>=0] for l in gt_labels_3d]

        bsz = len(points)# 获取批次大小（B）
        # 调用分割器处理点云，传入点云、元数据、真值框和标签，as_subsegmentor=True表示作为子模块运行
        seg_out_dict = self.segmentor(points=points, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, as_subsegmentor=True)
        # 提取分割特征
        seg_feats = seg_out_dict['seg_feats']#[225173, 131]
        # 若配置了detach_segmentor，将分割特征detach（切断梯度回传）
        if self.train_cfg.get('detach_segmentor', False):
            seg_feats = seg_feats.detach()
        seg_loss = seg_out_dict['losses']# 提取分割损失
        losses.update(seg_loss)# 将分割损失合并到总损失中

        dict_to_sample = dict(
            seg_points=seg_out_dict['seg_points'],# 分割后的点云 [225173, 5]
            seg_logits=seg_out_dict['seg_logits'].detach(),# 分割logits（detach避免梯度影响）[225173, 11]
            seg_vote_preds=seg_out_dict['seg_vote_preds'].detach(),# 投票预测（detach）[225173, 33]
            seg_feats=seg_feats,# 分割特征
            batch_idx=seg_out_dict['batch_idx'], # 点的批次索引
            vote_offsets=seg_out_dict['offsets'].detach(),# 投票偏移（detach）[225173, 33]
        )
        # 调用采样方法（sample/batched_group_sample等），按组采样前景点
        sampled_out = self.sample(dict_to_sample, dict_to_sample['vote_offsets'], gt_bboxes_3d, gt_labels_3d) # per cls list in sampled_out
        # 合并不同组的采样数据（如点云、特征、预测中心）
        combined_out = self.combine_classes(sampled_out, ['seg_points', 'seg_logits', 'seg_vote_preds', 'seg_feats', 'center_preds', 'batch_idx'])
        #将合并后的采样结果转换为适合检测头处理的特征（如体素特征）
        extract_output = self.extract_feat(combined_out, dict_to_sample, gt_bboxes_3d=gt_bboxes_3d, multiscale_features=seg_out_dict['decoder_features'])
        # 提取虚拟体素特征、坐标和中心（检测头的输入）
        voxel_feats = extract_output['virtual_feats']#[Vn,128]
        voxel_coors = extract_output['virtual_coors']#[Vn,4]
        voxel_xyz = extract_output['virtual_centers']#[Vn,3]
         # 检测头接收体素特征，输出预测结果
        outs = self.bbox_head(voxel_feats)
        # 准备损失计算的输入：预测结果、体素坐标、批次索引、真值、元数据
        loss_inputs = (outs['cls_logits'], outs['reg_preds']) + (voxel_xyz, voxel_coors[:, 0]) + (gt_bboxes_3d, gt_labels_3d, img_metas)
        # 计算检测损失（分类损失、回归损失等），支持IoU预测和辅助质心
        det_loss = self.bbox_head.loss(
            *loss_inputs, iou_logits=outs.get('iou_logits', None), gt_bboxes_ignore=gt_bboxes_ignore, aux_xyz=extract_output['virtual_centroid'])
        # 获取预测边界框及相关特征（用于后续精细处理或可视化）
        bbox_list = self.bbox_head.get_bboxes(
            outs['cls_logits'], outs['reg_preds'],
            voxel_xyz, voxel_feats, voxel_coors[:, 0], img_metas,
            rescale=False,
            iou_logits=outs.get('iou_logits', None))
        # 提取预测中心、类别、特征和预测框参数
        query_xyz = [x[0].gravity_center for x in bbox_list]#[VN,3]
        query_cat = [x[2] for x in bbox_list]#[VN]
        query_feats = [x[3] for x in bbox_list]#[VN,128]
        query_pred = [torch.cat([x[0].tensor[:, 3:], x[1][:, None]], dim=1) for x in bbox_list]#[VN,7]

        if hasattr(self.bbox_head, 'print_info'):
            self.print_info.update(self.bbox_head.print_info)
        losses.update(det_loss)# 将检测损失合并到总损失中
        losses.update(self.print_info)
        # 根据是否作为RPN（区域提议网络）返回不同格式的输出
        if self.as_rpn:
            output_dict = dict(
                rpn_losses=losses,
                cls_logits=outs['cls_logits'],
                reg_preds=outs['reg_preds'],
                voxel_xyz=voxel_xyz,
                voxel_batch_inds=voxel_coors[:, 0],
                pts_feats=extract_output['pts_feats'],
                pts_xyz=extract_output['pts_xyz'],
                pts_indicators=extract_output['pts_indicators'],
                pts_batch_inds=extract_output['pts_batch_inds'],
            )
            return output_dict
        else:
            output_dict = dict(
                losses=losses,
                cls_logits=outs['cls_logits'],
                reg_preds=outs['reg_preds'],
                voxel_feats=extract_output['out_voxel_feats'],
                voxel_coors=extract_output['out_coors'],
                voxel_xyz=extract_output['out_centers'],
                query_feats=query_feats,
                query_cat=query_cat,
                query_xyz=query_xyz,
                query_pred=query_pred,
            )
            return output_dict

    def combine_classes(self, data_dict, name_list):
        out_dict = {}
        for name in data_dict:
            if name in name_list:
                out_dict[name] = torch.cat(data_dict[name], 0) # 将该字段下的所有组的数据按第0维拼接（合并为一个张量）
        return out_dict

    def pre_voxelize(self, data_dict):
        raise NotImplementedError('No need to use prevoxelization anymore in FSDV2')
        batch_idx = data_dict['batch_idx']
        points = data_dict['seg_points']

        voxel_size = torch.tensor(self.cfg.pre_voxelization_size, device=batch_idx.device)
        pc_range = torch.tensor(self.cluster_assigner.point_cloud_range, device=points.device)
        coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        coors = coors[:, [2, 1, 0]] # to zyx order
        coors = torch.cat([batch_idx[:, None], coors], dim=1)

        new_coors, unq_inv  = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)

        voxelized_data_dict = {}
        for data_name in data_dict:
            data = data_dict[data_name]
            if data.dtype in (torch.float, torch.float16):
                voxelized_data, voxel_coors = scatter_v2(data, coors, mode='avg', return_inv=False, new_coors=new_coors, unq_inv=unq_inv)
                voxelized_data_dict[data_name] = voxelized_data

        voxelized_data_dict['batch_idx'] = voxel_coors[:, 0]
        return voxelized_data_dict


    def simple_test(self, points, img_metas, imgs=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None, return_res=False):
        """Test function without augmentaiton."""
        if gt_bboxes_3d is not None:
            gt_bboxes_3d = gt_bboxes_3d[0]
            gt_labels_3d = gt_labels_3d[0]
            assert isinstance(gt_bboxes_3d, list)
            assert isinstance(gt_labels_3d, list)
            assert len(gt_bboxes_3d) == len(gt_labels_3d) == 1, 'assuming single sample testing'

        bsz = len(points)

        seg_out_dict = self.segmentor.simple_test(points, img_metas, rescale=False)

        seg_feats = seg_out_dict['seg_feats']

        dict_to_sample = dict(
            seg_points=seg_out_dict['seg_points'],
            seg_logits=seg_out_dict['seg_logits'],
            seg_vote_preds=seg_out_dict['seg_vote_preds'],
            seg_feats=seg_feats,
            batch_idx=seg_out_dict['batch_idx'],
            vote_offsets = seg_out_dict['offsets']
        )
        sampled_out = self.sample(dict_to_sample, dict_to_sample['vote_offsets'], gt_bboxes_3d, gt_labels_3d) # per cls list in sampled_out

        combined_out = self.combine_classes(sampled_out, ['seg_points', 'seg_logits', 'seg_vote_preds', 'seg_feats', 'center_preds', 'batch_idx'])

        extract_output = self.extract_feat(combined_out, dict_to_sample, multiscale_features=seg_out_dict['decoder_features'])

        voxel_feats = extract_output['virtual_feats']
        voxel_coors = extract_output['virtual_coors']
        voxel_xyz = extract_output['virtual_centers']

        outs = self.bbox_head(voxel_feats)

        bbox_list = self.bbox_head.get_bboxes(
            outs['cls_logits'], outs['reg_preds'],
            voxel_xyz, voxel_feats, voxel_coors[:, 0], img_metas,
            rescale=rescale,
            iou_logits=outs.get('iou_logits', None))
        query_xyz = [x[0].gravity_center for x in bbox_list]
        query_cat = [x[2] for x in bbox_list]
        query_feats = [x[3] for x in bbox_list]
        query_pred = [torch.cat([x[0].tensor[:, 3:], x[1][:, None]], dim=1) for x in bbox_list]

        if return_res:
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels, _ in bbox_list
            ]
            bbox_results[0]['voxel_coors'] = extract_output['out_coors']
            return bbox_results

        if self.as_rpn:
            output_dict = dict(
                pts_feats=extract_output['pts_feats'],
                pts_xyz=extract_output['pts_xyz'],
                pts_indicators=extract_output['pts_indicators'],
                pts_batch_inds=extract_output['pts_batch_inds'],
                proposal_list=bbox_list
            )
            return output_dict
        else:
            output_dict = dict(
                cls_logits=outs['cls_logits'],
                reg_preds=outs['reg_preds'],
                voxel_feats=extract_output['out_voxel_feats'],
                voxel_coors=extract_output['out_coors'],
                voxel_xyz=extract_output['out_centers'],
                query_feats=query_feats,
                query_cat=query_cat,
                query_xyz=query_xyz,
                query_pred=query_pred,
            )
            return output_dict

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        return NotImplementedError


    def sample(self, dict_to_sample, offset, gt_bboxes_3d=None, gt_labels_3d=None):

        if self.cfg.get('group_sample', False):
            return self.group_sample(dict_to_sample, offset)

        if self.cfg.get('batched_group_sample', False):
            return self.batched_group_sample(dict_to_sample, offset)
        # 根据训练/测试模式选择配置
        cfg = self.train_cfg if self.training else self.test_cfg

        seg_logits = dict_to_sample['seg_logits']
        # 确保seg_logits未经过sigmoid激活（logits允许正负值）
        assert (seg_logits < 0).any() # make sure no sigmoid applied

        if seg_logits.size(1) == self.num_classes:
            seg_scores = seg_logits.sigmoid()# 对logits应用sigmoid得到[0,1]的分割分数
        else:
            raise NotImplementedError
        # 调整偏移量形状：(N, 类别数, 3)，3对应x/y/z坐标
        offset = offset.reshape(-1, self.num_classes, 3)
        seg_points = dict_to_sample['seg_points'][:, :3]# 提取点云的3D坐标（x/y/z）
        fg_mask_list = []# 存储每个类别的前景掩码（标记哪些点是该类的前景）
        center_preds_list = [] # 存储每个类别的预测中心（点坐标+偏移量）

        batch_idx = dict_to_sample['batch_idx'] # 每个点所属的样本索引（如batch中第0个、第1个样本）
        batch_size = batch_idx.max().item() + 1 # 计算batch中的样本数量
        for cls in range(self.num_classes):# 遍历每个类别
            cls_score_thr = cfg['score_thresh'][cls] # 该类别的分数阈值（用于筛选前景点）
            # 获取该类别的前景掩码：标记哪些点是该类的前景（可能结合分数、真实框等条件）
            fg_mask = self.get_fg_mask(seg_scores, seg_points, cls, batch_idx, gt_bboxes_3d, gt_labels_3d)
            # 确保每个样本至少有一个前景点（若不足，则强制选中每个样本的第一个点）
            if len(torch.unique(batch_idx[fg_mask])) < batch_size:
                # 获取每个样本的起始点位置（用于保底采样）
                one_random_pos_per_sample = self.get_sample_beg_position(batch_idx, fg_mask)
                fg_mask[one_random_pos_per_sample] = True # # 强制设为前景

            fg_mask_list.append(fg_mask)# 保存该类别的前景掩码
            # 计算该类别的预测中心：前景点坐标 + 对应类别的偏移量
            this_offset = offset[fg_mask, cls, :]# 提取前景点在该类别的偏移量
            this_points = seg_points[fg_mask, :]# 提取前景点的坐标
            this_centers = this_points + this_offset# 计算预测中心
            center_preds_list.append(this_centers)# 保存该类别的预测中心


        output_dict = {}
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask])# 按前景掩码提取每个类别的数据

            output_dict[data_name] = cls_data_list # 存储：{数据名: [类别0数据, 类别1数据, ...]}
        output_dict['fg_mask_list'] = fg_mask_list# 保存所有类别的前景掩码
        output_dict['center_preds'] = center_preds_list# 保存所有类别的预测中心

        return output_dict

    def get_sample_beg_position(self, batch_idx, fg_mask):
        assert batch_idx.shape == fg_mask.shape
        inner_inds = get_inner_win_inds(batch_idx.contiguous())
        pos = torch.where(inner_inds == 0)[0]
        return pos

    def get_fg_mask(self, seg_scores, seg_points, cls_id, batch_inds, gt_bboxes_3d, gt_labels_3d):
        if self.training and self.train_cfg.get('disable_pretrain', False) and not self.runtime_info.get('enable_detection', False):
            seg_scores = seg_scores[:, cls_id]# 提取当前组的分割分数：(总点数,)
            topks = self.train_cfg.get('disable_pretrain_topks', [100, 100, 100])# 每个类别的topk阈值
            k = min(topks[cls_id], len(seg_scores))# 取当前类别对应的k值（不超过总点数）
            top_inds = torch.topk(seg_scores, k)[1]# 取分数最高的k个点的索引
            fg_mask = torch.zeros_like(seg_scores, dtype=torch.bool) # 初始化掩码（全False）
            fg_mask[top_inds] = True # 标记topk个点为前景
        else:
            seg_scores = seg_scores[:, cls_id] # 提取当前类别的分割分数：(总点数,)
            cls_score_thr = self.cfg['score_thresh'][cls_id]# 当前类别的分数阈值（如0.5）
            if self.training and self.runtime_info is not None:
                buffer_thr = self.runtime_info.get('threshold_buffer', 0)# 训练模式下可能有阈值缓冲（用于动态调整阈值）
            else:
                buffer_thr = 0# 测试模式下无缓冲
            fg_mask = seg_scores > cls_score_thr + buffer_thr# 前景掩码：分数超过「阈值+缓冲」的点为前景

        # 条件：训练模式 + 配置启用「添加真实前景点」（add_gt_fg_points） + 未停止添加
        cfg = self.train_cfg if self.training else self.test_cfg

        if cfg.get('add_gt_fg_points', False) and self.training:
            if self.runtime_info.get('stop_add_gt_fg_points', False):
                return fg_mask
            bsz = len(gt_bboxes_3d)# batch大小（样本数量）
            # 断言：点数量、分数数量、批量索引数量一致
            assert len(seg_scores) == len(seg_points) == len(batch_inds)
            # 将点云按样本分割（每个样本的点单独处理）
            point_list = self.split_by_batch(seg_points, batch_inds, bsz)
            gt_fg_mask_list = []# 存储每个样本的「真实前景点掩码」

            for i, points in enumerate(point_list):
                # 找到当前样本中真实标签等于cls_id的边界框（筛选同类别的真实框）
                gt_mask = gt_labels_3d[i] == cls_id# 真实标签是否为当前类别
                gts = gt_bboxes_3d[i][gt_mask]# 筛选出的同类真实框
                # 若该样本无同类真实框，或无点，则掩码全False
                if not gt_mask.any() or len(points) == 0:
                    gt_fg_mask_list.append(gt_mask.new_zeros(len(points), dtype=torch.bool))
                    continue
                # 判断点是否在真实框内：in_box返回> -1表示在框内（视为前景）
                gt_fg_mask_list.append(gts.points_in_boxes(points) > -1)
            # 将每个样本的真实前景掩码合并为整体掩码（与batch_inds对应）
            gt_fg_mask = self.combine_by_batch(gt_fg_mask_list, batch_inds, bsz)
            fg_mask = fg_mask | gt_fg_mask# 合并：原前景掩码 或 真实前景掩码（取并集，补充真实框内的点）


        return fg_mask

    def split_by_batch(self, data, batch_idx, batch_size):
        assert batch_idx.max().item() + 1 <= batch_size
        data_list = []
        for i in range(batch_size):
            sample_mask = batch_idx == i
            data_list.append(data[sample_mask])
        return data_list

    def combine_by_batch(self, data_list, batch_idx, batch_size):
        assert len(data_list) == batch_size
        if data_list[0] is None:
            return None
        data_shape = (len(batch_idx),) + data_list[0].shape[1:]
        full_data = data_list[0].new_zeros(data_shape)
        for i, data in enumerate(data_list):
            sample_mask = batch_idx == i
            full_data[sample_mask] = data
        return full_data

    def group_sample(self, dict_to_sample, offset):
        bsz = dict_to_sample['batch_idx'].max().item() + 1
        assert bsz == 1, "Maybe some codes need to be modified if bsz > 1, this will be updated very soon"
        # combine all classes as fg class.
        cfg = self.train_cfg if self.training else self.test_cfg

        seg_logits = dict_to_sample['seg_logits']
        assert (seg_logits < 0).any() # make sure no sigmoid applied

        assert seg_logits.size(1) == self.num_classes + 1 # we have background class
        seg_scores = seg_logits.softmax(1)

        offset = offset.reshape(-1, self.num_classes + 1, 3)
        seg_points = dict_to_sample['seg_points'][:, :3]
        fg_mask_list = [] # fg_mask of each cls
        center_preds_list = [] # fg_mask of each cls


        cls_score_thrs = cfg['score_thresh']
        group_names = cfg['group_names']
        class_names = cfg['class_names']
        num_groups = len(group_names)
        assert num_groups == len(cls_score_thrs)
        assert isinstance(cls_score_thrs, (list, tuple))
        grouped_score = self.gather_group_by_names(seg_scores[:, :-1]) # without background score

        for i in range(num_groups):

            fg_mask = self.get_fg_mask(grouped_score, None, i, None, None, None)

            if not fg_mask.any():
                fg_mask[0] = True # at least one point

            fg_mask_list.append(fg_mask)

            tmp_idx = []
            for name in group_names[i]:
                tmp_idx.append(class_names.index(name))

            this_offset = offset[:, tmp_idx, :] 
            this_offset = this_offset[fg_mask, ...]
            this_logits = seg_logits[:, tmp_idx]
            this_logits = this_logits[fg_mask, :]

            offset_weight = self.get_offset_weight(this_logits)
            assert torch.isclose(offset_weight.sum(1), offset_weight.new_ones(len(offset_weight))).all()
            this_offset = (this_offset * offset_weight[:, :, None]).sum(dim=1)
            this_points = seg_points[fg_mask, :]
            this_centers = this_points + this_offset
            center_preds_list.append(this_centers)

        output_dict = {}
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask])

            output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = fg_mask_list
        output_dict['center_preds'] = center_preds_list

        return output_dict

    def batched_group_sample(self, dict_to_sample, offset):
        batch_idx = dict_to_sample['batch_idx']# 获取每个点所属的样本索引（区分不同样本）
        batch_size = batch_idx.max().item() + 1# 计算批量大小（样本数量）
        # # 根据训练/测试状态获取对应的配置
        cfg = self.train_cfg if self.training else self.test_cfg

        seg_logits = dict_to_sample['seg_logits']# 获取分割任务的logits（未归一化的预测值）
        # 断言：确保logits通道数 = 类别数 + 1（+1是背景类）
        assert seg_logits.size(1) == self.num_classes + 1 # we have background class
        seg_scores = seg_logits.softmax(1)# 对logits做softmax，得到每个类别的概率分数
        # 重塑偏移量：形状为 (点数量, 类别数+1, 3)，3对应x/y/z坐标偏移
        offset = offset.reshape(-1, self.num_classes + 1, 3)#[225173, 11, 3]
        seg_points = dict_to_sample['seg_points'][:, :3]# 获取点的坐标（取前3列，为x/y/z）
        # 初始化：存储每组的前景掩码和预测中心
        fg_mask_list = [] # 每个元素是一个布尔掩码，标记该组的前景点
        center_preds_list = [] # 每个元素是该组前景点的预测中心坐标


        cls_score_thrs = cfg['score_thresh']
        group_names = cfg['group_names']
        class_names = cfg['class_names']
        offset_scale = cfg.get('offset_scale', 1)
        num_groups = len(group_names)
        assert num_groups == len(cls_score_thrs)
        assert isinstance(cls_score_thrs, (list, tuple))
        # 按组聚合每个点云的概率分数（排除背景类，因为seg_scores[:, :-1]去掉了最后一列背景）
        grouped_score = self.gather_group_by_names(seg_scores[:, :-1]) # without background score[225173, 6]
        # 按组处理前景点
        for i in range(num_groups):
            # 获取当前组的前景掩码：筛选出分数高于该组阈值的点
            fg_mask = self.get_fg_mask(grouped_score, None, i, None, None, None)
            # 确保每个样本至少有一个前景点：如果某个样本没有前景点，随机选一个点补充
            if len(torch.unique(batch_idx[fg_mask])) < batch_size:
                one_random_pos_per_sample = self.get_sample_beg_position(batch_idx, fg_mask)
                fg_mask[one_random_pos_per_sample] = True # at least one point per sample

            fg_mask_list.append(fg_mask)

            tmp_idx = []# 获取当前组包含的类别在class_names中的索引
            for name in group_names[i]:
                tmp_idx.append(class_names.index(name))
            # 筛选当前组的偏移量（按前景掩码）
            this_offset = offset[:, tmp_idx, :] # 取当前组类别的偏移量
            this_offset = this_offset[fg_mask, ...]# 只保留前景点的偏移量[500,class_num,3]
            this_logits = seg_logits[:, tmp_idx]# 取当前组类别的logits
            this_logits = this_logits[fg_mask, :]# 只保留前景点的logits[500,class_num]

            offset_weight = self.get_offset_weight(this_logits)#计算偏移量的权重
            assert torch.isclose(offset_weight.sum(1), offset_weight.new_ones(len(offset_weight))).all()
            this_offset = (this_offset * offset_weight[:, :, None]).sum(dim=1)# 加权求和得到最终偏移量（组内类别偏移的加权平均）[500,3]
            this_points = seg_points[fg_mask, :]# 前景点的原始坐标[500,3]
            this_centers = this_points + offset_scale * this_offset#计算预测中心：原始点坐标 + 缩放后的偏移量
            center_preds_list.append(this_centers)# 保存当前组的预测中心

        output_dict = {}
        for data_name in dict_to_sample:# 对输入数据中的每个字段，按组筛选并保存
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask])# 用前景掩码筛选该组的数据

            output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = fg_mask_list
        output_dict['center_preds'] = center_preds_list

        return output_dict

    def get_offset_weight(self, seg_logit):
        mode = self.cfg['offset_weight']
        if mode == 'max':
            weight = ((seg_logit - seg_logit.max(1)[0][:, None]).abs() < 1e-6).float()
            assert ((weight == 1).any(1)).all()
            weight = weight / weight.sum(1)[:, None] # in case of two max values
            return weight
        else:
            raise NotImplementedError

    def gather_group(self, scores, group_lens):
        assert (scores >= 0).all()
        score_per_group = []
        beg = 0
        for group_len in group_lens:
            end = beg + group_len
            score_this_g = scores[:, beg:end].sum(1)
            score_per_group.append(score_this_g)
            beg = end
        assert end == scores.size(1) == sum(group_lens)
        gathered_score = torch.stack(score_per_group, dim=1)
        assert gathered_score.size(1) == len(group_lens)
        return  gathered_score

    def gather_group_by_names(self, scores):
        groups = self.cfg['group_names']
        class_names = self.cfg['class_names']
        assert (scores >= 0).all()
        score_per_group = []
        for g in groups:
            tmp_idx = []
            # 1. 找到当前组内所有类别在class_names中的索引
            for name in g:
                tmp_idx.append(class_names.index(name))
            # 2. 提取当前组所有类别的分数，并在类别维度求和
            # scores[:, tmp_idx]：取所有点在当前组类别上的分数（形状：[N, K]，N为点数量，K为组内类别数）
            # .sum(1)：在第1维（类别维度）求和，得到每个点在当前组的总分数（形状：[N]）
            score_per_group.append(scores[:, tmp_idx].sum(1))
        # 将每个组的分数堆叠成矩阵（形状：[N, M]，N为点数量，M为组数量）
        gathered_score = torch.stack(score_per_group, dim=1)
        return  gathered_score

    def get_batched_gt_fg_mask(self, points, batch_inds, gt_bboxes_3d):
        bsz = batch_inds.max().item() + 1
        point_list = self.split_by_batch(points, batch_inds, bsz)
        gt_fg_mask_list = []
        assert len(point_list) == len(gt_bboxes_3d)

        for i, points in enumerate(point_list):

            gts = gt_bboxes_3d[i]

            if len(gts) == 0 or len(points) == 0:
                gt_fg_mask_list.append(points.new_zeros(len(points), dtype=torch.bool))
                continue

            gt_fg_mask_list.append(gts.points_in_boxes(points) > -1)

        gt_fg_mask = self.combine_by_batch(gt_fg_mask_list, batch_inds, bsz)
        return gt_fg_mask