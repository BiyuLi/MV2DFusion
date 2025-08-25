
import warnings

import torch

from mmdet.core import bbox2result
from mmdet.models.detectors import SingleStageDetector, TwoStageDetector
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck


@DETECTORS.register_module()
class SingleStageDetectorWrapper(SingleStageDetector):
    def __init__(self,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward_train_w_feat(self,
                             feat,
                             img,
                             img_metas,
                             gt_bboxes,
                             gt_labels,
                             gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = feat
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        return losses

    def set_detection_cfg(self, detection_cfg):
        self.bbox_head.test_cfg = detection_cfg

    @torch.no_grad()
    def simple_test_w_feat(self, feat, img_metas, rescale=False):
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results


@DETECTORS.register_module()
class TwoStageDetectorWrapper(TwoStageDetector):
    def __init__(self,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward_train_w_feat(self,
                             feat,
                             img,
                             img_metas,
                             gt_bboxes,
                             gt_labels,
                             gt_bboxes_ignore=None,
                             gt_masks=None,
                             proposals=None,
                             **kwargs):
        x = feat#掩码后的特征，某些图像中可能没有真值直接被删掉了

        losses = dict()# 初始化空字典，用于存储所有计算出的损失

        # RPN forward and loss
        if self.with_rpn:# 判断模型是否包含RPN组件
            # 获取RPN生成候选框的配置（优先用训练配置，若无则用测试配置）
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            # 调用RPN头的训练前向方法，计算RPN损失并生成候选框
            rpn_losses, proposal_list = self.rpn_head.forward_train(#proposal_list=list[2000,5],多尺度特征图
                x,                              # 输入特征
                img_metas,                      # 图像元信息
                gt_bboxes,                      # 真实边界框（用于计算RPN损失）
                gt_labels=None,                  # RPN不依赖类别标签（仅负责生成候选框，不分类）
                gt_bboxes_ignore=gt_bboxes_ignore,  # 需要忽略的真实框
                proposal_cfg=proposal_cfg,# 候选框生成配置
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        # 对 RPN 生成的候选框进行精细化处理
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def set_detection_cfg(self, detection_cfg):
        self.roi_head.test_cfg = detection_cfg

    @torch.no_grad()
    def simple_test_w_feat(self, feat, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = feat
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
