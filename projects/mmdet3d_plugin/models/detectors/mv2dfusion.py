# Copyright (c) Wang, Z
# ------------------------------------------------------------------------
# Modified from StreamPETR (https://github.com/exiawsh/StreamPETR)
# Copyright (c) Shihao Wang
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS, build_detector, build_roi_extractor, build_neck, build_head
from mmdet.core import bbox2roi
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.builder import build_query_generator

import os
import cv2
import numpy as np
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from tools.debug_tools.visualize import vis_dets_2d
def bbox3d2result(bboxes, scores, labels, obj_idxes=None, track_scores=None, attrs=None):
    result_dict = dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu())

    if obj_idxes is not None:
        result_dict['track_ids'] = obj_idxes.cpu()
        result_dict['track_scores'] = track_scores.cpu()

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict


class GroupedItems(object):
    def __init__(self, items):
        self.items = items

    def __getitem__(self, key):
        return [x[key] for x in self.items]

    def __len__(self):
        return len(self.items)


@DETECTORS.register_module()
class MV2DFusion(MVXTwoStageDetector):
    """MV2D."""

    def __init__(self,
                 dataset='nuscenes',
                 # lidar branch
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 pts_backbone=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 pts_query_generator=None,
                 # image branch
                 img_backbone=None,
                 img_neck=None,
                 img_rpn_head=None,
                 img_roi_head=None,
                 img_roi_extractor=None,
                 img_query_generator=None,
                 use_grid_mask=False,
                 use_2d_proposal=False,
                 # fusion head
                 fusion_bbox_head=None,
                 # training
                 gt_mono_loss=False,
                 loss_weight_3d=1.,
                 loss_weight_pts=1.,
                 num_frame_head_grads=1,
                 num_frame_backbone_grads=1,
                 num_frame_losses=1,
                 # config
                 position_level=0,
                 test_clip_len=-1,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 debug=None,
                 ):
        self.dataset = dataset
        if pts_voxel_layer is not None:
            self.voxelize_reduce = pts_voxel_layer.pop('voxelize_reduce')

        super(MV2DFusion, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                                            pts_middle_encoder, pts_fusion_layer,
                                            img_backbone, pts_backbone, img_neck, pts_neck,
                                            pts_bbox_head, None, img_rpn_head,
                                            train_cfg, test_cfg, pretrained)

        if fusion_bbox_head is not None:
            fusion_train_cfg = train_cfg.fusion if train_cfg else None
            fusion_bbox_head.update(train_cfg=fusion_train_cfg)
            fusion_test_cfg = test_cfg.fusion if test_cfg else None
            fusion_bbox_head.update(test_cfg=fusion_test_cfg)
            self.fusion_bbox_head = build_head(fusion_bbox_head)

        if img_roi_head is not None:
            self.img_roi_head = build_detector(img_roi_head)
        if img_roi_extractor is not None:
            self.img_roi_extractor = build_roi_extractor(img_roi_extractor)
        if img_query_generator is not None:
            img_query_generator.update(dict(loss_cls=self.fusion_bbox_head.loss_cls))
            self.img_query_generator = build_query_generator(img_query_generator)

        if pts_query_generator is not None:
            pts_query_generator.update(dict(
                dataset=dataset,
                virtual_voxel_size=self.pts_backbone.virtual_voxel_size,
                point_cloud_range=self.pts_backbone.point_cloud_range,
                head_pc_range=self.fusion_bbox_head.pc_range.tolist(),
            ))
            self.pts_query_generator = build_query_generator(pts_query_generator)

        self.use_2d_proposal = use_2d_proposal
        self.gt_mono_loss = gt_mono_loss

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.position_level = position_level

        self.loss_weight_3d = loss_weight_3d
        self.loss_weight_pts = loss_weight_pts

        self.test_clip_len = test_clip_len
        self.test_clip_id = 1

        self.current_seq = 0

        self.debug = debug

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img = img.squeeze(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)# 主干网络输出多尺度特征
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())# 若为字典，转换为列表（便于后续处理）
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)# 颈部网络（FPN）融合特征，将主干输出的多尺度特征融合为统一通道（256 维）的特征图，平衡语义信息和细节信息。
        img_feats_det = img_feats# 保存融合后的特征，用于后续检测（如RPN、RoI头）
        # 获取指定层级的特征（self.position_level=2，对应中间层特征）
        BN, C, H, W = img_feats[self.position_level].size()
        if self.training or training_mode:
            # 训练模式：重塑为 [B, len_queue, N_cam, C, H, W][1, 1, 6, 256, 40, 100]
            img_feats_reshaped = img_feats[self.position_level].view(B, len_queue, int(BN / B / len_queue), C, H, W)
        else:
            # 推理模式：无需时序维度，重塑为 [B, N_cam, C, H, W]
            img_feats_reshaped = img_feats[self.position_level].view(B, int(BN / B / len_queue), C, H, W)
        return img_feats_reshaped, img_feats_det

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=[], out_fp32=True)
    def extract_pts_feat(self, points):
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
            x = self.pts_middle_encoder(feats, coords, batch_size)
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x

    def obtain_history_memory(self,
                              gt_bboxes_3d=None,
                              gt_labels_3d=None,
                              gt_bboxes=None,
                              gt_labels=None,
                              img_metas=None,
                              centers2d=None,
                              depths=None,
                              gt_bboxes_ignore=None,
                              **data):
        losses = dict() # 存储所有帧的损失
        T = data['img'].size(1)# 获取时序帧数（data['img']维度通常为[B, T, ...]，T为帧数）
        # 计算无需计算梯度的帧数和无需返回损失的帧数
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):# 遍历每一帧（i为帧索引，0表示最早帧，T-1表示最新帧）
            requires_grad = False# 是否计算梯度（默认不计算）
            return_losses = False# 是否返回损失（默认不返回）
            data_t = dict()# 存储当前帧i的数据
            # 1. 提取当前帧i的数据（从多帧数据中拆分）
            for key in data:
                if key in ['instance_inds_2d', 'points', 'pts_feats']:
                    data_t[key] = data[key][i]# 这些key的数据维度为[B, T, ...]，直接取第i帧
                elif key in ['proposals']:
                    data_t[key] = data[key][i]# 候选框数据取第i帧
                else:# 其他数据（如图像特征）维度为[B, T, ...]，取第i列（按帧维度索引）
                    data_t[key] = data[key][:, i]
            # 2. 处理当前帧的图像特征
            data_t['img_feats'] = data_t['img_feats']# 从当前帧数据中提取图像特征
            # 3. 决定是否计算梯度（仅最新的self.num_frame_head_grads帧计算）
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            # 5. 调用点云训练前向函数，计算当前帧的损失
            loss = self.forward_pts_train(gt_bboxes_3d[i],
                                          gt_labels_3d[i], gt_bboxes[i],
                                          gt_labels[i], img_metas[i], centers2d[i], depths[i],
                                          requires_grad=requires_grad, return_losses=return_losses, **data_t)
            if loss is not None:# 6. 收集损失（仅当return_losses=True时有效）
                for key, value in loss.items():
                    losses['frame_' + str(i) + "_" + key] = value
        return losses

    def prepare_detection_data(self, img_metas, **data):
        ori_imgs = imgs = data['img']
        if imgs.dim() == 5:
            B, V, C, H, W = imgs.shape
        else:
            B, V, C, H, W = 1, *imgs.shape
        imgs = imgs.flatten(0, 1)
        feats = [x.flatten(0, 1) for x in data['img_feats_for_det']]#5个尺度的特征图
        img_metas_det = [dict() for _ in range(B * V)]

        for b in range(B):
            for v in range(V):
                img_meta = {
                    'img_shape': img_metas[b]['img_shape'][v],# 当前视图预处理后的形状
                    'ori_shape': img_metas[b]['ori_shape'][:3],# 原始图像的形状（取前3维，可能含通道）
                    'pad_shape': img_metas[b]['pad_shape'][v],# 填充后的形状（预处理时可能补边）
                    'batch_input_shape': (H, W),# 批量输入的统一形状（高、宽）
                    'scale_factor': img_metas[b]['scale_factor'],# 图像缩放因子（原始到预处理的缩放比例）
                    'intrinsics': data['intrinsics'][b][v],# 相机内参矩阵（用于3D到2D投影）
                    'extrinsics': data['extrinsics'][b][v],# 相机外参矩阵（用于坐标系转换）
                    'lidar2img': data['lidar2img'][b][v],# 激光雷达到图像的转换矩阵（融合点云与图像时用）
                    'num_views': V,# 总视图数量（当前样本的视图数）
                    # for debug
                    'img': ori_imgs[b, v],
                    'img_norm_cfg': img_metas[b]['img_norm_cfg'],
                    'scene_token': img_metas[b]['scene_token'],
                    'filename': img_metas[b]['filename'][v],
                }
                # 添加前一帧存在标志（可能用于时序模型）
                img_meta['prev_exists'] = data['prev_exists'][b].clone()
                # 训练时添加标注信息（监督信号）
                if self.training:
                    if 'instance_inds_2d' in data:
                        instance_inds_2d = data['instance_inds_2d'][b][v].clone()
                        img_meta['instance_inds'] = instance_inds_2d

                    img_meta['gt_bboxes'] = data['gt_bboxes'][b][v].clone()
                    img_meta['gt_labels'] = data['gt_labels'][b][v].clone()

                img_metas_det[b * V + v] = img_meta
        return imgs, feats, img_metas_det, (B, V, C, H, W)

    def convert_to_fsd_anno(self, boxes, labels, inv=False):
        if len(labels) == 0:
            return boxes, labels.long()

        b_tensor = boxes.tensor.clone()# 复制边界框的 tensor 数据（避免修改原始数据）
        if b_tensor.size(1) == 9:
            vel = b_tensor[:, -2:]# 若边界框有9个参数，最后2个视为速度信息（vel）
            b_tensor = b_tensor[:, :-2]# 分离出核心边界框参数（前7个）
        else:
            vel = None# 无速度信息时为 None

        if self.dataset == 'nuscenes':
            tgt_cls = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                       'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
            fsd_cls = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                       'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
        elif self.dataset == 'argov2':
            tgt_cls = ['ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 'BOX_TRUCK', 'BUS',
                       'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 'DOG', 'LARGE_VEHICLE',
                       'MESSAGE_BOARD_TRAILER', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE',
                       'MOTORCYCLIST', 'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN',
                       'STOP_SIGN', 'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
                       'WHEELCHAIR', 'WHEELED_DEVICE', 'WHEELED_RIDER']
            fsd_cls = ['Regular_vehicle', 'Pedestrian', 'Bicyclist', 'Motorcyclist', 'Wheeled_rider', 'Bollard',
                       'Construction_cone', 'Sign', 'Construction_barrel', 'Stop_sign',
                       'Mobile_pedestrian_crossing_sign', 'Large_vehicle', 'Bus', 'Box_truck', 'Truck',
                       'Vehicular_trailer', 'Truck_cab', 'School_bus', 'Articulated_bus', 'Message_board_trailer',
                       'Bicycle', 'Motorcycle', 'Wheeled_device', 'Wheelchair', 'Stroller', 'Dog']
        else:
            raise NotImplementedError

        tgt_cls = [x.lower() for x in tgt_cls] # 原始类别名称转为小写（避免大小写匹配问题）
        fsd_cls = [x.lower() for x in fsd_cls] # FSD 类别名称转为小写
        # 创建“原始类别 -> FSD 类别”的索引映射表
        to_fsd_cls_map = [fsd_cls.index(x) for x in tgt_cls]
        # 创建“FSD 类别 -> 原始类别”的索引映射表（用于反向转换）
        to_tgt_cls_map = [tgt_cls.index(x) for x in fsd_cls]
        if not inv:
            # 正向转换（原始 -> FSD）
            # 调整角度参数（第6列，可能是3D框的朝向角）
            b_tensor[:, 6] = -b_tensor[:, 6] - np.pi / 2
            # 选择“原始 -> FSD”的类别映射表
            cls_map = labels.new_tensor(to_fsd_cls_map)
            # 重排边界框参数的列顺序（可能对应不同格式的参数定义）
            b_tensor = b_tensor[:, [0, 1, 2, 4, 3, 5, 6]]
        else:
            # 反向转换（FSD -> 原始）
            b_tensor[:, 6] = -(b_tensor[:, 6] + np.pi / 2)
            cls_map = labels.new_tensor(to_tgt_cls_map)
            b_tensor = b_tensor[:, [0, 1, 2, 4, 3, 5, 6]]
        if vel is not None:
             # 若有速度信息，将其拼接回边界框 tensor（保持9个参数）
            b_tensor = torch.cat([b_tensor, vel], dim=1)
        # 重构 boxes 对象（保持原 boxes 类的属性和方法）
        boxes = boxes.__class__(b_tensor, box_dim=b_tensor.size(-1))
        # 映射类别标签（通过映射表转换索引）
        labels = cls_map[labels]
        return boxes, labels

    @auto_fp16(apply_to=('imgs', 'feats'))
    def forward_roi_head(self, imgs, feats, img_metas):
        dets2d = self.img_roi_head.simple_test_w_feat(feats, img_metas)
        dets = self.process_2d_detections(dets2d, imgs.device)#得到每个相机的所有检测结果
        return dets2d, dets

    @auto_fp16(apply_to=('imgs', 'feats'))
    def forward_roi_head_train(self, imgs, feats, img_metas, gt_bboxes, gt_labels):
        # TODO: check 2d annotation
        gt_bboxes = sum(gt_bboxes, [])#统一不同批次的标注格式，便于后续逐样本处理。
        gt_labels = sum(gt_labels, [])
        valid_inds = imgs.new_zeros(len(gt_bboxes), dtype=torch.bool)
        gt_bboxes_valid, gt_labels_valid, img_metas_valid = [], [], []
        #用于存储筛选后的有效标注（gt_bboxes_valid、gt_labels_valid）和对应的图像元数据（img_metas_valid）
        for i in range(len(gt_bboxes)):
            if len(gt_bboxes[i]) > 0:
                gt_bboxes_valid.append(gt_bboxes[i])
                gt_labels_valid.append(gt_labels[i])
                img_metas_valid.append(img_metas[i])
                valid_inds[i] = 1

        if not valid_inds.any():# 当所有样本都无有效标注时
            # grad for all parameters
            # 手动创建虚拟标注（避免训练中断）
            gt_bboxes_valid = [imgs.new_tensor([[40, 120, 40, 120]])]# 虚拟边界框（坐标示例）
            gt_labels_valid = [imgs.new_tensor([0], dtype=torch.int64)] # 虚拟标签（类别0）
            # 调用ROI Head的训练接口，但仅使用第一个样本的特征和图像
            losses = self.img_roi_head.forward_train_w_feat(
                [x[:1] for x in feats], imgs[:1], img_metas[:1], gt_bboxes_valid, gt_labels_valid, )
             # 将所有损失值清零（虚拟标注不贡献梯度）
            losses = {k: ([x * 0 for x in v] if isinstance(v, (list, tuple)) else v * 0) for k, v in losses.items()}
            # 处理可能的NaN值（替换为0，保证训练稳定）
            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and v.isnan().any():
                    losses[k] = v.nan_to_num()
        else:
            # 筛选有效样本的特征、图像和元数据，调用ROI Head计算损失
            losses = self.img_roi_head.forward_train_w_feat(
                [x[valid_inds] for x in feats], imgs[valid_inds], img_metas_valid, gt_bboxes_valid, gt_labels_valid, )
        return losses

    @staticmethod
    def box_iou(rois_a, rois_b, eps=1e-4):
        rois_a = rois_a[..., None, :]  # [*, n, 1, 4]
        rois_b = rois_b[..., None, :, :]  # [*, 1, m, 4]
        xy_start = torch.maximum(rois_a[..., 0:2], rois_b[..., 0:2])
        xy_end = torch.minimum(rois_a[..., 2:4], rois_b[..., 2:4])
        wh = torch.maximum(xy_end - xy_start, rois_a.new_tensor(0))  # [*, n, m, 2]
        intersect = wh.prod(-1)  # [*, n, m]
        wh_a = rois_a[..., 2:4] - rois_a[..., 0:2]  # [*, m, 1, 2]
        wh_b = rois_b[..., 2:4] - rois_b[..., 0:2]  # [*, 1, n, 2]
        area_a = wh_a.prod(-1)
        area_b = wh_b.prod(-1)
        union = area_a + area_b - intersect
        iou = intersect / (union + eps)
        return iou

    def process_2d_gt(self, gt_bboxes, gt_labels, device):
        return [torch.cat(
            [bboxes.to(device), torch.ones([len(labels), 1], dtype=bboxes.dtype, device=device),
             labels.unsqueeze(-1).to(bboxes.dtype)], dim=-1).to(device)
                for bboxes, labels in zip(gt_bboxes, gt_labels)]
    #核心功能是将检测结果中未覆盖到的真值框（ground truth boxes）补充到检测结果中，
    # 确保最终结果既包含模型预测的检测框，也包含那些未被检测到但确实存在的真实目标框。
    def complement_2d_gt(self, detections, gts, thr=0.6, with_id=False):
        # detections: [n, 6], gts: [m, 6]
        if len(detections) == 0:
            if len(gts) == 0:
                gts = gts.new_zeros([0, 6]) # 若真值也为空，返回空tensor
            if with_id:# 若需要ID，给真值框增加一列ID（值为-1，通常表示无跟踪ID）
                gts = torch.cat([gts, torch.zeros_like(gts[..., :1]) - 1], dim=-1)
            return gts # 检测结果为空时，直接返回真值框（作为补充）
        if len(gts) == 0:
            return detections# 无真值框时，直接返回检测结果（无需补充）
        iou = self.box_iou(gts, detections)# 计算真值框与检测框的IOU，形状为 [m, n]
        max_iou = iou.max(-1)[0] # 对每个真值框，取与所有检测框的最大IOU，形状为 [m,]
        complement_ids = max_iou <= thr# 筛选出最大IOU ≤ 阈值的真值框（未被检测覆盖）
        min_bbox_size = self.img_roi_head.test_cfg.get('min_bbox_size', 0)
        wh = gts[:, 2:4] - gts[:, 0:2]
        valid_ids = (wh >= min_bbox_size).all(dim=1) # 筛选宽和高均 ≥ 最小尺寸的真值框
        complement_gts = gts[complement_ids & valid_ids]# 同时满足“未被覆盖”和“尺寸有效”的真值框
        if with_id:
            complement_gts = torch.cat([complement_gts, torch.zeros_like(complement_gts[..., :1]) - 1], dim=-1)
        return torch.cat([detections, complement_gts], dim=0)
    #主要功能是将检测结果将 “按类别拆分的检测框列表” 转换为 “包含类别 ID 的统一检测框张量”，并过滤掉过小的边界框，以便后续处理（如 3D 检测或跟踪）。
    def process_2d_detections(self, results, device):
        """
        :param results:
            results: list[per_cls_res] of size BATCH_SIZE
            per_cls_res: list(boxes) of size NUM_CLASSES
            boxes: ndarray of shape [num_boxes, 5->(x1, y1, x2, y2, score)]
        :return:
            detections: list[ndarray of shape [num_boxes, 6->(x1, y1, x2, y2, score, label_id)]] of size len(results)
        """
        detections = [torch.cat(
            [torch.cat([
                torch.tensor(boxes, device=device),
                torch.full((len(boxes), 1), label_id, dtype=torch.float, device=device)], dim=1)
                # if len(boxes) > 0 else torch.zeros((0, 6), device=device)
                for label_id, boxes in enumerate(res)], dim=0) for res in results]
        min_bbox_size = self.img_roi_head.test_cfg.get('min_bbox_size', 0)
        #过滤过小的边界框
        if min_bbox_size > 0:
            new_detections = []
            for det in detections:
                wh = det[:, 2:4] - det[:, 0:2]
                valid = (wh >= min_bbox_size).all(dim=1)
                new_detections.append(det[valid])
            detections = new_detections
        return detections

    def extract_roi_feats(self, feats_det, rois, **data):
        roi_feats = self.img_roi_extractor(feats_det[:self.img_roi_extractor.num_inputs], rois)
        return roi_feats

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):                                                                                                                                                                                                                                                                                                                                                                                                                     
        data['gt_bboxes'] = gt_bboxes
        data['gt_labels'] = gt_labels
        data['gt_bboxes_3d'] = gt_bboxes_3d
        data['gt_labels_3d'] = gt_labels_3d
        data['depths'] = depths
        #[6, 3, 640, 1600]、多尺度特征图、每个batch中每个相机对应的元数据、原始图像尺寸
        imgs_det, feats_det, img_metas_det, imgs_shape = self.prepare_detection_data(img_metas, **data)

        B, V = imgs_shape[:2]
        if not requires_grad:
            raise NotImplementedError
        else:
            # image query generation
            if self.with_img_roi_head:#2D 检测头前向传播
                losses_det2d = self.forward_roi_head_train(imgs_det, feats_det, img_metas_det, gt_bboxes, gt_labels)

            self.eval()
            with torch.no_grad():
                #生成单批次下多相机的2D 检测结果dets(V,N,6) 6=X1,Y1,X2,Y2,socre,class_id
                dets2d, dets = self.forward_roi_head(imgs_det, feats_det, img_metas_det)
                #处理 2D 真实标注，得到dets_gt（用于补充检测结果）
                dets_gt = self.process_2d_gt(sum(gt_bboxes, []), sum(gt_labels, []), imgs_det.device)
                assert len(dets) == (len(dets_gt))
                if self.use_2d_proposal:
                    dets = sum([x for x in data['proposals']], [])
                dets = [self.complement_2d_gt(det, det_gt, )#将真实标注补充到检测结果中（可能用于提升训练稳定性）
                        for det, det_gt in zip(dets, dets_gt)]

                #防止检测结果为空：若dets为空，手动添加一个虚拟候选框
                if sum([len(p) for p in dets]) == 0:
                    proposal = torch.tensor([[10, 20, 30, 40, 0, 1]], dtype=dets[0].dtype, device=dets[0].device)
                    dets = [proposal] + dets[1:]
                #将边界框（dets）转换为 ROI（Region of Interest）格式，便于后续提取 ROI 特征。
                rois = bbox2roi(dets)#[N,5]
            self.train()
            #从图像特征（feats_det）中提取 ROI 区域的特征（roi_feats）[93, 256, 7, 7]
            roi_feats = self.extract_roi_feats(feats_det, rois, **data)
            #计算每个相机的ROI 数量（n_rois_per_view/n_rois_per_batch），用于特征分组
            n_rois_per_view = [len(p) for p in dets]
            n_rois_per_batch = [sum(n_rois_per_view[i * V: (i + 1) * V]) for i in range(B)]
            #基于 ROI 特征生成动态查询（dyn_query）和辅助特征（dyn_feats），这些查询将用于后续与点云特征的融合
            #dyn_query[N,50,4] 4=雷达坐标系下的位置＋深度概率。
            dyn_query, dyn_feats = self.img_query_generator(roi_feats, dets, img_metas_det,
                                                            n_rois_per_view=n_rois_per_view,
                                                            n_rois_per_batch=n_rois_per_batch,
                                                            data=data)
            dyn_feats_pred = dyn_feats

            if self.gt_mono_loss:# 单目深度损失处理
                rois_gt = bbox2roi(dets_gt)
                roi_feats_gt = self.extract_roi_feats(feats_det, rois_gt, **data)
                n_rois_per_view_gt = [len(p) for p in dets_gt]
                n_rois_per_batch_gt = [sum(n_rois_per_view_gt[i * V: (i + 1) * V]) for i in range(B)]
                _, dyn_feats = self.img_query_generator(roi_feats_gt, dets_gt, img_metas_det,
                                                        n_rois_per_view=n_rois_per_view_gt,
                                                        n_rois_per_batch=n_rois_per_batch_gt,
                                                        data=data,)
                n_rois_per_batch = n_rois_per_batch_gt #93

            # lidar query generation
            # 转换3D标注为FSDet格式
            fsd_gt_bboxes_3d, fsd_gt_labels_3d = [], []
            for b in range(B):
                box, label = self.convert_to_fsd_anno(gt_bboxes_3d[b], gt_labels_3d[b])
                fsd_gt_bboxes_3d.append(box)
                fsd_gt_labels_3d.append(label)
            # 点云 backbone 前向传播
            out_dict = self.pts_backbone.forward_train(data['pts_feats'], img_metas, fsd_gt_bboxes_3d, fsd_gt_labels_3d)
            # 生成点云查询特征
            pts_feat, pts_pos, pts_query_feat, pts_query_center = self.pts_query_generator(
                out_dict['voxel_feats'], out_dict['voxel_coors'], out_dict['voxel_xyz'], out_dict['query_feats'],
                out_dict['query_xyz'], out_dict['query_pred'], out_dict['query_cat'], B)
            losses_pts = out_dict['losses']

            outs = self.fusion_bbox_head(img_metas, dyn_query=dyn_query, dyn_feats=dyn_feats_pred,
                                      pts_query_center=pts_query_center, pts_query_feat=pts_query_feat,
                                      pts_feat=pts_feat, pts_pos=pts_pos, pts_shape=None, **data)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]

            # fusion head loss
            losses = self.fusion_bbox_head.loss(*loss_inputs)

            # point cloud detector loss
            for k, v in losses_pts.items():
                if 'loss' in k:
                    v = v * self.loss_weight_pts
                losses['pts.' + k] = v

            # image detector loss
            if self.with_img_roi_head:
                for k, v in losses_det2d.items():
                    losses['det2d.' + k] = v

            # image query generator auxiliary loss
            if dyn_feats is not None:
                losses_img_qg = dict()

                # monodepth loss
                if 'd_loss' in dyn_feats:
                    losses_img_qg['d_loss'] = dyn_feats['d_loss']

                # query generator loss
                if dyn_feats.get('cls_scores', None) is not None:
                    cls_scores = dyn_feats['cls_scores'].split(n_rois_per_batch, dim=0)
                    bbox_preds = dyn_feats['bbox_preds'].split(n_rois_per_batch, dim=0)
                    gt_bboxes_3d_ = [torch.cat((b.gravity_center, b.tensor[:, 3:]),
                                               dim=1).to(imgs_det.device).clone() for b in gt_bboxes_3d]
                    for x in gt_bboxes_3d_:
                        x[:, 6:] = 0
                    loss_cls, loss_bbox = self.fusion_bbox_head.loss_single(cls_scores, bbox_preds, gt_bboxes_3d_,
                                                                         gt_labels_3d)
                    losses_img_qg.update({'loss_cls': loss_cls, 'loss_bbox': loss_bbox})

                for k, v in losses_img_qg.items():
                    losses['imgqg.' + k] = v

            # loss scaling
            for k, v in losses.items():
                if 'loss' in k:
                    if 'det2d.' not in k:
                        losses[k] = v * self.loss_weight_3d

            return losses
        else:
            return None

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        B, T, V, _, H, W = data['img'].shape
        # 拆分历史帧（不计算主干网络梯度）和近期帧（计算主干网络梯度）
        prev_img = data['img'][:, :-self.num_frame_backbone_grads]# 历史帧：取前(T - K)帧
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]# 近期帧：取最后K帧（需计算梯度）
        
        rec_img_feats, rec_img_feats_for_det = self.extract_img_feat(rec_img, self.num_frame_backbone_grads)
        # 若存在历史帧（T - K > 0），则提取其特征
        if T - self.num_frame_backbone_grads > 0:
            self.eval()# 模型设为推理模式（关闭BN层更新、Dropout等）
            with torch.no_grad():# 禁用梯度计算（历史帧不参与参数更新）
                prev_img_feats, prev_img_feats_for_det = self.extract_img_feat(prev_img,
                                                                               T - self.num_frame_backbone_grads, True)
            self.train() # 恢复训练模式（仅影响后续操作）
            data['img_feats'] = torch.cat([prev_img_feats, rec_img_feats], dim=1)
            # 处理用于检测头的多尺度特征（按尺度分别拼接）
            prev_T = T - self.num_frame_backbone_grads
            rec_T = self.num_frame_backbone_grads
            data['img_feats_for_det'] = [# 每个尺度的特征：拼接历史帧和近期帧（维度调整为[B, T, V, C, H, W]）
                torch.cat([prev.view(B, prev_T, V, *prev.shape[1:]), rec.view(B, rec_T, V, *rec.shape[1:])], dim=1)
                for prev, rec in zip(prev_img_feats_for_det, rec_img_feats_for_det)]
            data['img_feats_for_det'] = GroupedItems(data['img_feats_for_det'])# 封装为GroupedItems（方便多尺度特征处理）
        else:
            # 若没有历史帧（T <= K），直接使用近期帧特征
            data['img_feats'] = rec_img_feats
            data['img_feats_for_det'] = GroupedItems([x.view(B, T, V, *x.shape[1:]) for x in rec_img_feats_for_det])
        data['pts_feats'] = data['points']

        losses = self.obtain_history_memory(gt_bboxes_3d,
                                            gt_labels_3d, gt_bboxes,
                                            gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data)

        return losses
    def map_global_img_indices(self, global_indices, dets):
        """
        把 flatten 之后的全局索引映射到 (cam_id, local_idx)
        Args:
            global_indices: List[int] 或 Tensor，flatten 后的一维索引
            dets: List[Tensor]，长度=相机数，每个 shape=(Ni,6)
        Returns:
            List[Tuple[int, int]]，其中 int 是 (cam_id, local_idx)
        """
        mapping = []
        start = 0
        for cam_id, cam_dets in enumerate(dets):
            num = len(cam_dets)
            end = start + num
            for gidx in global_indices:
                if start <= gidx < end:
                    local_idx = gidx - start
                    mapping.append((cam_id, local_idx))
            start = end
        return mapping
    def vis_bev_segmentation_style(self,
                               voxel_xyz,
                               query_pred,
                               query_cat,
                               bboxes,
                               scores,
                               labels,
                               highlight_idx=None,
                               mask=None,
                               bev_range=[-50, 50, -50, 50],
                               out_size=(512, 512)):
        """
        可视化 BEV 分割和 query 高亮（高亮目标为红色）
        
        Args:
            voxel_xyz: [N_voxels, 3] 点云中心
            query_pred: [N_queries, 3] query 位置
            query_cat: [N_queries] 类别
            bboxes, scores, labels: 最终3D检测结果
            highlight_idx: 需要高亮的 query idx
            mask: [N_queries] 布尔数组，False 的 query 不绘制
            bev_range: [x_min, x_max, y_min, y_max]
            out_size: 输出图像大小
        Returns:
            bev_img: [H, W, 3] 可视化 BEV 图
        """
        import numpy as np
        import cv2
        import matplotlib.cm as cm

        H, W = out_size
        bev_img = np.zeros((H, W, 3), dtype=np.uint8)

        # 确保 tensor 在 CPU 并转换为 numpy
        voxel_xyz = voxel_xyz.detach().cpu().numpy()
        query_pred = query_pred[0].detach().cpu().numpy()
        query_cat = query_cat[0].detach().cpu().numpy()
        if mask is not None:
            mask = mask.detach().cpu().numpy()

        x_min, x_max, y_min, y_max = bev_range

        # 映射点云到 BEV 灰色背景
        xs = ((voxel_xyz[:, 0] - x_min) / (x_max - x_min) * W).astype(np.int32)
        ys = ((y_max - voxel_xyz[:, 1]) / (y_max - y_min) * H).astype(np.int32)
        xs = np.clip(xs, 0, W - 1)
        ys = np.clip(ys, 0, H - 1)
        bev_img[ys, xs, :] = 200  # 灰色点云背景

        # 设置类别 colormap
        cmap = cm.get_cmap('tab20', np.max(query_cat) + 1)

        # 绘制每个 query
        for i, (x, y, _) in enumerate(query_pred):
            if mask is not None and not mask[i]:
                continue  # 被 mask 掉的 query 不绘制
            cx = int((x - x_min) / (x_max - x_min) * W)
            cy = int((y_max - y) / (y_max - y_min) * H)
            color = (np.array(cmap(query_cat[i])[:3]) * 255).astype(np.uint8)
            if highlight_idx is not None and i == highlight_idx:
                # 高亮 query: 红色填充 + 白色边框
                cv2.circle(bev_img, (cx, cy), 10, (0, 0, 255), -1)
                cv2.circle(bev_img, (cx, cy), 12, (255, 255, 255), 2)
            else:
                # 普通 query
                cv2.circle(bev_img, (cx, cy), 4, color.tolist(), -1)

        # 绘制 3D 检测框
        for bbox, score, label in zip(bboxes, scores, labels):
            x, y, z = bbox[:3]
            cx = int((x - x_min) / (x_max - x_min) * W)
            cy = int((y_max - y) / (y_max - y_min) * H)
            if highlight_idx is not None:
                # 高亮框用红色
                cv2.rectangle(bev_img, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 0, 255), 2)
            else:
                cv2.rectangle(bev_img, (cx - 3, cy - 3), (cx + 3, cy + 3), (0, 128, 255), 1)

        return bev_img


    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""

        if (img_metas[0]['scene_token'] != self.prev_scene_token) or (
                self.test_clip_len > 0 and self.test_clip_id % self.test_clip_len == 0):
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.fusion_bbox_head.reset_memory()
            self.test_clip_id = 1
            self.current_seq += 1
        else:
            data['prev_exists'] = data['img'].new_ones(1)
            self.test_clip_id += 1

        imgs_det, feats_det, img_metas_det, imgs_shape = self.prepare_detection_data(img_metas, **data)
        # feats_det: feats_det[level] = [n_cam, feat_c, lvl_h, lvl_w]
        B, V = imgs_shape[:2]

        # image query generation
        dets2d, dets = self.forward_roi_head(imgs_det, feats_det, img_metas_det)
        # dets: dets[cam_idx] = [n_dets, 6]  6: [x1, x2, y1, y2, score, cls]
        
        # debug: vis roi detections
        if True:
            from tools.debug_tools.visualize import vis_dets_2d
            vis_dets_2d(img_metas_det, dets, 0.5, "./vis_det2d_0.5")
        if self.use_2d_proposal:
            dets = data['proposals']

        if sum([len(p) for p in dets]) == 0:
            proposal = torch.tensor([[0, 50, 50, 100, 0, 1]], dtype=dets[0].dtype,
                                    device=dets[0].device)
            dets = [proposal] + dets[1:]
        rois = bbox2roi(dets)  #rois: [img_idx, x1, y1, x2, y2]

        roi_feats = self.extract_roi_feats(feats_det, rois, **data) # roi_feats: [n_rois, 256, 7, 7]
        n_rois_per_view = [len(p) for p in dets]
        n_rois_per_batch = [sum(n_rois_per_view[i * V: (i + 1) * V]) for i in range(B)]
        dyn_query, dyn_feats = self.img_query_generator(roi_feats, dets, img_metas_det,
                                                    n_rois_per_view=n_rois_per_view,
                                                    n_rois_per_batch=n_rois_per_batch,
                                                    data=dict())
        #dyn_query[batch]:(n_rois, n_depth_bins, 4) -> (x, y, z, depth_prob), 表示预计的在某个深度bin目标的x,y,z lidar坐标及在这个深度的置信度 
        # lidar query generation
        out_dict = self.pts_backbone.simple_test(data['pts_feats'], img_metas)
        pts_feat, pts_pos, pts_query_feat, pts_query_center = self.pts_query_generator(
            out_dict['voxel_feats'], out_dict['voxel_coors'], out_dict['voxel_xyz'], out_dict['query_feats'],
            out_dict['query_xyz'], out_dict['query_pred'], out_dict['query_cat'], B)

        outs = self.fusion_bbox_head(img_metas, dyn_query=dyn_query, dyn_feats=dyn_feats,
                                  pts_query_center=pts_query_center, pts_query_feat=pts_query_feat, pts_feat=pts_feat,
                                  pts_pos=pts_pos, pts_shape=None, **data)

        lidar2img_dict = outs['lidar2img_dict']   # {lidar_idx: [img_query_indices]}
        bevmask=outs['tgt_query_mask']
        save_dir = "./vis_lidar_img_pairs_filter_ronhe_assignment"
       
        
        # 先获取 3D 检测结果
        bbox_list = self.fusion_bbox_head.get_bboxes(outs, img_metas)
        bboxes, scores, labels = bbox_list[0]  # batch=0
        from tools.debug_tools.visualize import vis_dets_2d, vis_bev
        import os
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import cm
        os.makedirs(save_dir, exist_ok=True)
        # 遍历 lidar2img_dict
        for lidar_idx, img_indices in lidar2img_dict.items():
            # 左边 BEV 图
            bev_img = self.vis_bev_segmentation_style(
                            voxel_xyz=out_dict['voxel_xyz'],
                            query_pred=out_dict['query_xyz'],
                            query_cat=out_dict['query_cat'],
                            bboxes=bboxes,
                            scores=scores,
                            labels=labels,
                            highlight_idx=lidar_idx,
                            mask=bevmask,
                            bev_range=[-50,50,-50,50],
                            out_size=(512,512)
                        )
            # 构造每个相机的选中框
            filtered_dets = []
            for cam_id, cam_dets in enumerate(dets):
                # 找出属于该相机的 topk 图像query框
                selected = [local_idx for (cid, local_idx) in self.map_global_img_indices(img_indices, dets)
                            if cid == cam_id]
                if len(selected) > 0:
                    selected_tensor = torch.tensor(selected, device=cam_dets.device, dtype=torch.long)
                    cam_dets_selected = cam_dets[selected_tensor]
                else:
                    cam_dets_selected = cam_dets.new_zeros((0, 6))
                filtered_dets.append(cam_dets_selected)

            # 右边 相机拼图
            cam_img = vis_dets_2d(img_metas_det, filtered_dets, 0.0, return_img=True)

            # =========================
            # 调整 BEV 和相机图比例
            bev_h, bev_w, _ = bev_img.shape
            cam_h, cam_w, _ = cam_img.shape

            # 统一高度
            target_h = max(bev_h, cam_h)

            # BEV 缩放
            scale_bev = target_h / bev_h
            bev_w_new = int(bev_w * scale_bev)
            bev_img_resized = cv2.resize(bev_img, (bev_w_new, target_h))

            # 相机图缩放
            scale_cam = target_h / cam_h
            cam_w_new = int(cam_w * scale_cam)
            cam_img_resized = cv2.resize(cam_img, (cam_w_new, target_h))

            # 拼接
            canvas = np.ones((target_h, bev_w_new + cam_w_new, 3), dtype=np.uint8) * 255
            canvas[:, :bev_w_new] = bev_img_resized
            canvas[:, bev_w_new:bev_w_new + cam_w_new] = cam_img_resized

            # 保存
            out_path = os.path.join(save_dir, f"lidar_{lidar_idx}.jpg")
            cv2.imwrite(out_path, canvas)
            print(f"[INFO] 保存到 {out_path}")
        
        # =========================
        # img_index=outs['top2_img_indices']
        # selected_boxes, mappings = self.get_2d_boxes_from_indices(dets, img_index)
        # # 构造每个相机的选中框
        # filtered_dets = []
        # for cam_id, cam_dets in enumerate(dets):
        #     selected = [local_idx for (cid, local_idx) in mappings if cid == cam_id]
        #     if len(selected) > 0:
        #         selected_tensor = torch.tensor(selected, device=cam_dets.device, dtype=torch.long)
        #         cam_dets_selected = cam_dets[selected_tensor]
        #     else:
        #         cam_dets_selected = cam_dets.new_zeros((0, 6))
        #     filtered_dets.append(cam_dets_selected)

        # # 一次性拼成 2×3 大图
        # save_dir = "./vis_top5"
        # from tools.debug_tools.visualize import vis_dets_2d
        # vis_dets_2d(img_metas_det, filtered_dets, 0.4, save_dir)
        
        
        bbox_results = [
            bbox3d2result(*bbox)
            for bbox in bbox_list
        ]
        return bbox_results
    
    def get_2d_boxes_from_indices(self,dets, img_index):
        """
        dets: list of length V, dets[v] is [N,6]
        img_index: 1D tensor of indices after flattening all dets
        return: list of selected 2D boxes [M, 6], and (cam_id, local_idx) mapping
        """
        device = img_index.device
        n_per_view = [len(d) for d in dets]       
        cum_sum = torch.cumsum(torch.tensor([0] + n_per_view, device=device), dim=0)  # 保证在同一设备
        
        boxes = []
        mappings = []
        for idx in img_index:
            # idx 是 tensor -> 转成 int 之前要先挪到 cpu
            cam_id = (cum_sum[1:] > idx).nonzero(as_tuple=False)[0].item()
            local_idx = idx.item() - cum_sum[cam_id].item()
            
            box = dets[cam_id][local_idx]  # [6]
            boxes.append(box.unsqueeze(0))
            mappings.append((cam_id, local_idx))
        
        return torch.cat(boxes, dim=0), mappings


    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        B, V, _, H, W = data['img'].shape
        img_feats_reshaped, img_feats_for_det = self.extract_img_feat(data['img'], 1)
        data['img_feats'] = img_feats_reshaped
        data['img_feats_for_det'] = [x.view(B, V, *x.shape[1:]) for x in img_feats_for_det]

        rec_points = [data['points'].squeeze(0)]
        data['pts_feats'] = rec_points

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

    def forward_test(self, img_metas, rescale, **data):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))

        for key in data:
            if key in ['instance_inds_2d']:
                data[key] = data[key][0][0]
            elif key in ['proposals']:
                data[key] = data[key][0][0]
            elif key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:  # key == 'img'
                data[key] = data[key][0]

        for i in range(len(img_metas[0])):
            img_metas[0][i]['lidar2img'] = data['lidar2img'][i].cpu().numpy()
            img_metas[0][i]['intrinsics'] = data['intrinsics'][i].cpu().numpy()
            img_metas[0][i]['extrinsics'] = data['extrinsics'][i].cpu().numpy()
        results = self.simple_test(img_metas[0], **data)
        return results

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        for key in ['proposals', 'instance_inds_2d', 'points']:
            if key in data:
                data[key] = list(zip(*data[key]))

        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                if key in data:
                    data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)