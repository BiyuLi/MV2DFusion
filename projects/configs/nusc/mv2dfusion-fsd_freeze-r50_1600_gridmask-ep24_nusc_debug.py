_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/default_runtime.py',
]
plugin = True
plugin_dir = [
    'projects/mmdet3d_plugin/',
    'projects/fsdv2/'
]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# FSDv2 setting
voxel_size = [0.2, 0.2, 8]  # 体素尺寸（点云分辨率）
point_cloud_range = [-54.4, -54.4, -5.0, 54.4, 54.4, 3.0]  # 点云范围（x,y,z的最小值和最大值）
sparse_shape = [40, 544, 544]  # 稀疏体素网格维度
target_sparse_shape = [20, 272, 272]  # 目标稀疏形状（融合后特征尺寸）
fsd_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
seg_voxel_size = (0.2, 0.2, 0.2)  # 分割体素尺寸
virtual_voxel_size = (0.4, 0.4, 0.4)  # 虚拟体素尺寸（稀疏区域补充特征）
group1 = ['car']  # 类别分组1
group2 = ['truck', 'construction_vehicle']  # 类别分组2
group3 = ['bus', 'trailer']  # 类别分组3
group4 = ['barrier']  # 类别分组4
group5 = ['motorcycle', 'bicycle']  # 类别分组5
group6 = ['pedestrian', 'traffic_cone']  # 类别分组6
group_names = [group1, group2, group3, group4, group5, group6]  # 所有类别分组
seg_score_thresh = [0.2, ] * 3 + [0.1, ] * 3  # 分割分数阈值（前3组用0.2，后3组用0.1）
group_lens = [len(group1), len(group2), len(group3), len(group4), len(group5), len(group6)]  # 每组类别数量
head_group1 = fsd_class_names[:5]  # 检测头部分组1（前5类）
head_group2 = fsd_class_names[5:]  # 检测头部分组2（后5类）
tasks = [
    dict(class_names=head_group1),  # 子任务1
    dict(class_names=head_group2),  # 子任务2
]

# training hyperparameter
num_gpus = 1  # GPU数量
batch_size = 1  # 每个GPU的批量大小
num_iters_per_epoch = 28130 // (num_gpus * batch_size)  # 每个epoch的迭代次数
num_epochs = 24  # 总训练epoch数
queue_length = 1  # 数据队列长度
num_frame_losses = 1  # 参与损失计算的帧数（时序信息利用）

pts_ckpt = 'weights/fsdv2-converted.pth'  # 点云模型预训练权重路径
img_ckpt = 'weights/mask_rcnn_r50_fpn_1x_nuim_20201008_195238-e99f5182.pth'  # 图像模型预训练权重路径

roi_size = 7  # ROI区域大小
roi_strides = [4, 8, 16, 32, 64]  # 特征图步长（对应不同尺度特征图的下采样率）
model = dict(
    type='MV2DFusion',  # 模型类型：多模态融合
    dataset='nuscenes',  # 数据集名称
    num_frame_head_grads=num_frame_losses,  # 参与头部网络梯度计算的帧数
    num_frame_backbone_grads=num_frame_losses,  # 参与主干网络梯度计算的帧数
    num_frame_losses=num_frame_losses,  # 参与损失计算的帧数（时序信息利用）
    position_level=2,  # 位置编码的层级（通过编码增强特征的空间位置信息，提升定位精度）
    use_grid_mask=True,  # 启用网格掩码（在特征图上生成局部掩码，强制模型关注重要区域，抑制背景噪声）

    loss_weight_3d=0.1,  # 3D检测损失的权重
    loss_weight_pts=1.,  # 点云相关损失的权重（点级损失更重要）
    gt_mono_loss=True,  # 是否使用基于真值的单目损失（通过真值监督单目视图特征学习，缓解单目深度估计模糊问题）

    img_backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=img_ckpt,
            prefix='backbone.', map_location='cpu'),
        type='ResNet',  # 主干网络类型：ResNet
        depth=50,  # 网络深度：50层
        num_stages=4,  # 网络阶段数
        out_indices=(0, 1, 2, 3),  # 输出的特征层索引（多尺度特征，满足不同大小目标的检测需求）
        frozen_stages=-1,  # 冻结的网络阶段（-1表示不冻结）
        norm_cfg=dict(type='BN2d', requires_grad=False),  # 归一化配置
        norm_eval=True,  # 推理模式下冻结BN层
        with_cp=True,  # 是否使用checkpointing（节省内存，适合大模型训练）
        style='pytorch'),  # 网络风格：PyTorch
    img_neck=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=img_ckpt,
            prefix='neck.', map_location='cpu'),
        type='FPN',  # 颈部网络类型：特征金字塔网络（FPN）
        in_channels=[256, 512, 1024, 2048],  # 输入特征通道（对应ResNet的4个输出层）
        out_channels=256,  # 输出特征通道（统一多尺度特征的通道数）
        num_outs=5),  # 输出的特征层数量（生成5个不同尺度的特征图）
    img_roi_extractor=dict(
        type='SingleRoIExtractor',  # ROI提取器类型：单ROI提取器
        roi_layer=dict(type='RoIAlign', output_size=roi_size, sampling_ratio=-1),  # ROI对齐层（精确提取感兴趣区域特征）
        featmap_strides=roi_strides[:-1],  # 特征图步长（对应不同尺度特征图的下采样率）
        out_channels=256, ),  # 输出特征通道
    # faster rcnn
    img_roi_head=dict(
        type='TwoStageDetectorWrapper',  # 两阶段检测器包装器
        init_cfg=dict(
            type='Pretrained', checkpoint=img_ckpt,
            map_location='cpu'),
        rpn_head=dict(
            type='RPNHead',  # 候选框生成网络
            in_channels=256,  # 输入特征通道（FPN输出）
            feat_channels=256,  # 特征通道数
            anchor_generator=dict(  # 锚点生成器（预设候选框）
                type='AnchorGenerator',
                scales=[8],  # 锚点尺度
                ratios=[0.5, 1.0, 2.0],  # 锚点宽高比
                strides=roi_strides),  # 不同特征图上的锚点步长
            bbox_coder=dict(  # 边界框编码器（将坐标差转换为回归目标）
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],  # 均值
                target_stds=[1.0, 1.0, 1.0, 1.0]),  # 标准差
            loss_cls=dict(  # 分类损失（判断锚点是否为目标）
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),  # 边界框回归损失
        ),
        roi_head=dict(
            type='StandardRoIHead',  # 标准RoI头
            bbox_roi_extractor=dict(  # ROI提取器
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=roi_strides[:-1]),
            bbox_head=dict(  # 边界框头
                type='Shared2FCBBoxHead',  # 共享2个全连接层的边界框头
                in_channels=256,  # 输入特征通道
                fc_out_channels=1024,  # 全连接层输出通道
                roi_feat_size=7,  # ROI特征大小（7x7）
                num_classes=10,  # 目标类别数（nuScenes数据集有10类目标）
                bbox_coder=dict(  # 边界框编码器
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),  # 第一阶段：较大标准差
                reg_class_agnostic=False,  # 回归与类别相关（每类目标单独回归参数）
                loss_cls=dict(  # 分类损失
                    type='FocalLoss',  # Focal损失解决类别不平衡
                    use_sigmoid=True,
                    gamma=2.0,  # 难样本加权系数
                    alpha=0.25,  # 正负样本权重比
                    loss_weight=1.0),
                reg_decoded_bbox=True,  # 基于解码后的边界框计算损失（更直观）
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)  # 边界框损失（GIoU更鲁棒）
            ),
        ),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(  # 锚点与GT的匹配规则
                    type='MaxIoUAssigner',  # 基于IoU的最大匹配策略
                    pos_iou_thr=0.7,  # IoU≥0.7的锚点视为正样本（目标）
                    neg_iou_thr=0.3,  # IoU≤0.3的锚点视为负样本（背景）
                    min_pos_iou=0.3,  # 正样本的最小IoU阈值
                    match_low_quality=True,  # 允许低质量匹配
                    ignore_iof_thr=-1),  # 忽略与GT的IoF阈值
                sampler=dict(  # 样本采样策略
                    type='RandomSampler',
                    num=256,  # 每批采样256个样本
                    pos_fraction=0.5,  # 正样本占比50%
                    neg_pos_ub=-1,  # 负样本相对正样本的最大比例
                    add_gt_as_proposals=False),  # 不将GT直接作为候选框
                allowed_border=-1,  # 允许候选框超出图像边界的范围
                pos_weight=-1,  # 正样本的损失权重
                debug=False),  # 是否输出调试信息
            rpn_proposal=dict(  # RPN候选框生成配置
                nms_pre=2000,  # NMS前保留的候选框数量
                max_per_img=1000,  # 单张图像最终保留的候选框上限
                nms=dict(type='nms', iou_threshold=0.7),  # NMS算法
                min_bbox_size=0),  # 候选框的最小尺寸
            rcnn=dict(  # RCNN阶段配置
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            min_bbox_size=4,  # 过滤尺寸小于4的候选框
            rpn=dict(
                nms_pre=1000,  # 推理时NMS前保留1000个候选框
                max_per_img=1000,  # 单图RPN输出上限
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,  # 过滤分数<0.05的框
                nms=dict(type='nms', iou_threshold=0.6, class_agnostic=True),  # 跨类别NMS
                max_per_img=60,)),  # 单图最终输出60个检测结果
    ),
    img_query_generator=dict(
        type='ImageDistributionQueryGenerator',  # 图像分布查询生成器
        prob_bin=25,  # 概率分箱数（将连续概率离散化为25个区间）
        depth_range=[0.1, 90],  # 深度范围（图像中目标可能的距离）
        gt_guided=False,  # 不使用GT引导查询生成（纯模型预测）
        gt_guided_loss=1.,  # GT引导损失权重
        with_cls=True,  # 生成包含类别信息的查询
        with_size=True,  # 生成包含尺寸信息的查询

        with_avg_pool=True,  # 使用全局平均池化
        num_shared_convs=1,  # 共享卷积层数量
        num_shared_fcs=1,  # 共享全连接层数量
        in_channels=256,  # 输入特征通道（来自FPN）
        fc_out_channels=1024,  # 全连接层输出通道
        roi_feat_size=roi_size,  # ROI特征大小
        extra_encoding=dict(  # 额外编码（如相机内参）
            num_layers=2,  # 编码层数
            feat_channels=[512, 256],  # 编码通道数
            features=[dict(type='intrinsic', in_channels=16, )]  # 处理16维相机内参特征
        ),
    ),
    # fsdv2
    pts_backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=pts_ckpt, prefix='pts_backbone.', map_location='cpu'),
        type='SingleStageFSDV2',  # 单阶段全稀疏检测架构
        freeze=True,  # 冻结部分参数（可能用于预训练权重固定）
        norm_eval=True,  # 推理模式下冻结BN层
        segmentor=dict(  # 点云分割器（生成点级语义特征）
            type='VoteSegmentor',  # 带投票机制的点云分割器
            tanh_dims=[],  # 不使用tanh激活的维度
            voxel_layer=dict(  # 体素化层：将点云划分为体素
                voxel_size=seg_voxel_size,  # 体素尺寸
                max_num_points=-1,  # 每个体素最大点数（-1表示无限制）
                point_cloud_range=point_cloud_range,  # 点云范围
                max_voxels=(-1, -1)  # 最大体素数量（-1表示无限制）
            ),
            voxel_encoder=dict(  # 体素编码器：将体素内点云编码为体素特征
                type='DynamicScatterVFE',  # 动态聚集体素编码器
                in_channels=5,  # 输入点云维度（x,y,z,intensity,时间戳或其他特征）
                feat_channels=[64, 64],  # 两层卷积通道数（逐步升维）
                voxel_size=seg_voxel_size,  # 体素尺寸
                with_cluster_center=True,  # 编码体素内点的聚类中心
                with_voxel_center=True,  # 编码体素中心坐标
                point_cloud_range=point_cloud_range,  # 点云范围
                norm_cfg=dict(type='MyBN1d', eps=1e-3, momentum=0.01),  # 自定义1D BatchNorm
                unique_once=True,  # 避免重复计算体素特征
            ),
            middle_encoder=dict(
                type='PseudoMiddleEncoderForSpconvFSD',  # 适配稀疏卷积的中间层
            ),
            backbone=dict(  # 稀疏主干网络：提取多尺度体素特征
                type='SimpleSparseUNet',  # 稀疏U-Net（编码-解码结构，保留多尺度特征）
                in_channels=64,  # 输入特征通道
                sparse_shape=sparse_shape,  # 稀疏体素网格维度
                order=('conv', 'norm', 'act'),  # 卷积→归一化→激活的操作顺序
                norm_cfg=dict(type='MyBN1d', eps=1e-3, momentum=0.01),
                base_channels=64,  # 基础通道数
                output_channels=128,  # 输出通道数（dummy）
                encoder_channels=((128,), (128, 128,), (128, 128,), (128, 128, 128), (256, 256, 256), (256, 256, 256)),  # 编码器各层通道数
                encoder_paddings=((1,), (1, 1,), (1, 1,), (1, 1, 1), (1, 1, 1), (1, 1, 1)),  # 编码器填充
                decoder_channels=((256, 256, 256), (256, 256, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128)),  # 解码器各层通道数
                decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1), (1, 1)),  # 解码器填充
                return_multiscale_features=True,  # 输出多尺度特征
            ),
            decode_neck=dict(  # 解码颈部：将体素特征映射回点云特征
                type='Voxel2PointScatterNeck',  # 体素到点云的散射映射
                voxel_size=seg_voxel_size,  # 体素尺寸
                point_cloud_range=point_cloud_range,  # 点云范围
            ),
            segmentation_head=dict(  # 分割头：输出点云语义分割结果+投票损失
                type='VoteSegHead',  # 带投票损失的分割头
                in_channel=67 + 64,  # 输入通道（67维点特征+64维体素特征）
                hidden_dims=[128, 128],  # 两层隐藏层
                num_classes=len(class_names),  # 目标类别数
                dropout_ratio=0.0,  # Dropout比率
                conv_cfg=dict(type='Conv1d'),  # 卷积配置
                norm_cfg=dict(type='MyBN1d'),  # 归一化配置
                act_cfg=dict(type='ReLU'),  # 激活函数配置
                loss_decode=dict(  # 语义分割损失
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    class_weight=[1.0, ] * len(class_names) + [0.1,],  # 背景类权重降低
                    loss_weight=10.0),  # 分割损失权重
                loss_vote=dict(  # 投票损失
                    type='L1Loss',
                    loss_weight=1.0),
            ),
            train_cfg=dict(
                point_loss=True,  # 是否启用点级损失
                score_thresh=seg_score_thresh,  # 分割分数阈值
                class_names=fsd_class_names,  # 类别名称
                group_names=group_names,  # 类别分组名称
                group_lens=group_lens,  # 每组类别数量
            ),
        ),
        virtual_point_projector=dict(
            in_channels=83 + 64,  # 输入特征通道（83维原始特征+64维编码特征）
            hidden_dims=[64, 64],  # 两层MLP（特征降维与增强）
            norm_cfg=dict(type='MyBN1d'),  # 归一化配置

            ori_in_channels=67 + 64,  # 原始点特征输入通道
            ori_hidden_dims=[64, 64],  # 原始特征处理MLP

            # TODO: optional
            recover_in_channels=128 + 3,  # 恢复特征输入（128维特征+3维偏移）
            recover_hidden_dims=[128, 128],  # 恢复特征MLP（精细化虚拟点特征）
        ),
        multiscale_cfg=dict(
            multiscale_levels=[0, 1, 2],  # 融合第0/1/2层多尺度特征
            projector_hiddens=[[256, 128], [128, 128], [128, 128]],  # 各层特征投影器（降维融合）
            fusion_mode='avg',  # 融合方式为平均（简单有效，避免过拟合）
            target_sparse_shape=target_sparse_shape,  # 目标稀疏形状（融合后特征尺寸）
            norm_cfg=dict(type='MyBN1d'),  # 归一化配置
        ),
        voxel_encoder=dict(
            type='DynamicScatterVFE',  # 动态聚集体素编码器
            in_channels=67,  # 输入通道（虚拟点特征维度）
            feat_channels=[64, 128],  # 特征升维（64→128）
            voxel_size=virtual_voxel_size,  # 虚拟体素尺寸（比真实体素大，降低计算量）
            with_cluster_center=True,  # 编码体素内点的聚类中心
            with_voxel_center=True,  # 编码体素中心坐标
            point_cloud_range=point_cloud_range,  # 点云范围
            norm_cfg=dict(type='MyBN1d', eps=1e-3, momentum=0.01),  # 自定义1D BatchNorm
            unique_once=True,  # 避免重复计算体素特征
        ),
        backbone=dict(
            type='VirtualVoxelMixer',  # 虚拟体素混合网络（融合真实与虚拟体素特征）
            in_channels=128,  # 输入通道（来自虚拟体素编码器）
            sparse_shape=target_sparse_shape,  # 目标稀疏形状
            order=('conv', 'norm', 'act'),  # 卷积→归一化→激活的操作顺序
            norm_cfg=dict(type='MyBN1d', eps=1e-3, momentum=0.01),  # 归一化配置
            base_channels=64,  # 基础通道数
            output_channels=128,  # 输出通道数
            encoder_channels=((64,), (64, 64,), (64, 64,),),  # 编码器通道（逐步提取高层特征）
            encoder_paddings=((1,), (1, 1,), (1, 1,),),  # 编码器填充
            decoder_channels=((64, 64, 64), (64, 64, 64), (64, 64, 64)),  # 解码器通道（恢复细节特征）
            decoder_paddings=((1, 1), (1, 1), (1, 1),),  # 解码器填充
        ),
        bbox_head=dict(
            type='FSDV2Head',  # 点云检测头
            as_rpn=False,  # 是否作为RPN使用
            num_classes=len(class_names),  # 目标类别数
            bbox_coder=dict(type='BasePointBBoxCoder', code_size=10),  # 10维边界框编码（3D位置+3D尺寸+旋转+速度）
            loss_cls=dict(
                type='FocalLoss',  # 分类损失（Focal损失解决类别不平衡）
                use_sigmoid=True,
                gamma=2.0,  # 难样本加权系数
                alpha=0.25,  # 正负样本权重比
                loss_weight=4.0),  # 分类损失权重
            loss_center=dict(type='L1Loss', loss_weight=0.5),  # 中心坐标回归损失
            loss_size=dict(type='L1Loss', loss_weight=0.5),  # 尺寸回归损失
            loss_rot=dict(type='L1Loss', loss_weight=0.2),  # 旋转角回归损失
            loss_vel=dict(type='L1Loss', loss_weight=0.2),  # 速度回归损失
            in_channel=128,  # 输入特征通道
            shared_mlp_dims=[256, 256],  # 共享MLP（特征细化）
            train_cfg=None,  # 训练配置
            test_cfg=None,  # 测试配置
            norm_cfg=dict(type='MyBN1d'),  # 归一化配置
            tasks=tasks,  # 分任务检测（前5类/后5类）
            class_names=fsd_class_names,  # 类别名称
            common_attrs=dict(  # 目标属性预测配置
                center=(3, 2, 128), dim=(3, 2, 128), rot=(2, 2, 128), vel=(2, 2, 128)
                # (out_dim, num_layers, hidden_dim)
            ),
            num_cls_layer=2,  # 分类层数
            cls_hidden_dim=128,  # 分类隐藏层通道数
            separate_head=dict(  # 分头部处理不同任务（避免任务干扰）
                type='FSDSeparateHead',
                norm_cfg=dict(type='MyBN1d'),  # 归一化配置
                act='relu',  # 激活函数
            ),
        ),
        train_cfg=dict(
            score_thresh=seg_score_thresh,  # 分割分数阈值
            sync_reg_avg_factor=True,  # 基于目标质心分配样本（更精准匹配）
            batched_group_sample=True,  # 按类别组批量采样（平衡各类样本）
            offset_weight='max',  # 偏移权重
            class_names=fsd_class_names,  # 类别名称
            group_names=[group1, group2, group3, group4, group5, group6],  # 类别分组名称
            centroid_assign=True,  # 是否启用质心分配
            disable_pretrain=True,  # 禁用预训练
            disable_pretrain_topks=[500, ] * len(class_names),  # 禁用预训练的Top-K样本
        ),
        test_cfg=dict(
            score_thresh=seg_score_thresh,  # 分割分数阈值
            batched_group_sample=True,  # 按类别组批量采样
            offset_weight='max',  # 偏移权重
            class_names=fsd_class_names,  # 类别名称
            group_names=[group1, group2, group3, group4, group5, group6],  # 类别分组名称
            use_rotate_nms=True,  # 旋转NMS（适应3D目标方向差异）
            nms_pre=-1,  # NMS前保留的候选框数量
            nms_thr=0.25,  # NMS IoU阈值（过滤重复框）
            score_thr=0.05,  # 分数阈值
            min_bbox_size=0,  # 最小边界框尺寸
            max_num=500,  # 单类最大输出500个框
            all_task_max_num=200,  # 全任务最大输出200个框（控制冗余）
        ),
    ),
    pts_query_generator=dict(
        type='PointCloudQueryGenerator',
        in_channels=128,
        hidden_channel=128,
        pts_use_cat=False,
    ),
    # fusion head
    fusion_bbox_head=dict(
        type='MV2DFusionHead',  # 融合头部
        prob_bin=25,  # 概率分箱数（将连续概率分布离散化为25个区间，用于特征编码）
        post_bev_nms_ops=[0],  # BEV（鸟瞰图）后处理的NMS操作索引（[0]表示仅执行一次NMS）
        num_classes=10,  # 目标类别数
        in_channels=256,  # 输入特征通道数（图像与点云特征融合后的通道维度）
        num_query=300,  # 生成300个查询向量（潜在目标）
        memory_len=6 * 256,  # 特征记忆长度（用于时序特征缓存，6×256=1536，适配多帧特征）
        topk_proposals=256,  # 从候选框中筛选Top-256个高置信度框用于融合
        num_propagated=256,  # 传播到下一层的特征数量（保持特征传播效率）
        with_ego_pos=True,  # 融合时引入ego车辆位置信息（补偿车辆自身运动对检测的影响）
        match_with_velo=True,  # 匹配时考虑目标速度信息（提升动态目标检测精度）
        scalar=10,  # 噪声分组系数（控制训练时添加的噪声强度，增强模型泛化性）
        noise_scale=1.0,  # 噪声缩放因子（调节噪声幅度）
        dn_weight=1.0,  # 去噪训练（DN，Denoising Training）的损失权重（增强查询向量抗干扰能力）
        split=0.75,  # 正样本比例（训练时正样本占比75%，平衡正负样本）
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 3D边界框各维度的回归权重
        transformer=dict(
            type='MV2DFusionTransformer',  # 多模态融合Transformer
            decoder=dict(
                type='MV2DFusionTransformerDecoder',  # 融合解码器
                return_intermediate=True,  # 返回中间层结果（用于辅助训练或集成预测）
                num_layers=6,  # 解码器层数（6层逐步优化特征，平衡精度与效率）
                transformerlayers=dict(  # 每层解码器的结构
                    type='MV2DFusionTransformerDecoderLayer',
                    attn_cfgs=[  # 注意力配置（自注意力+交叉注意力）
                        dict(
                            type='MultiheadAttention',  # 多头自注意力
                            embed_dims=256,  # 嵌入维度
                            num_heads=8,  # 注意力头数
                            dropout=0.1),  # Dropout比率
                        dict(
                            type='MixedCrossAttention',  # 混合交叉注意力
                            embed_dims=256,  # 嵌入维度
                            num_groups=8,  # 特征分组数（与注意力头数匹配）
                            num_levels=4,  # 融合的特征层级（多尺度特征融合）
                            num_cams=6,  # 相机数量（nuScenes数据集通常用6个相机）
                            dropout=0.1,  # Dropout比率
                            num_pts=13,  # 点云采样点数（每个目标采样13个点特征）
                            bias=2.,  # 注意力偏置（增强重要特征权重）
                            attn_cfg=dict(
                                type='PETRMultiheadFlashAttention',  # 高效Flash注意力（加速计算）
                                batch_first=False,  # 批次优先
                                embed_dims=256,  # 嵌入维度
                                num_heads=8,  # 注意力头数
                                dropout=0.1),),  # Dropout比率
                    ],
                    feedforward_channels=2048,  # 前馈网络通道数（特征映射能力，2048=256×8）
                    ffn_dropout=0.1,  # 前馈网络Dropout比率
                    with_cp=True,  # 使用checkpointing（节省内存，适合大模型训练）
                    operation_order=(  # 操作顺序：自注意力→归一化→交叉注意力→归一化→前馈网络→归一化
                        'self_attn', 'norm',
                        'cross_attn', 'norm',
                        'ffn', 'norm')
                ),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',  # 无NMS编码器（替代传统NMS，避免阈值敏感问题）
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],  # 后处理中心范围
            pc_range=point_cloud_range,  # 点云范围
            max_num=300,  # 最大输出框数量（与num_query一致，避免冗余）
            voxel_size=voxel_size,  # 体素尺寸（用于坐标映射）
            num_classes=10),  # 目标类别数
        loss_cls=dict(
            type='FocalLoss',  # 分类损失（Focal损失解决类别不平衡）
            use_sigmoid=True,  # 使用Sigmoid激活
            gamma=2.0,  # 难样本加权系数（增大难样本损失）
            alpha=0.25,  # 正负样本权重比（正样本占25%）
            loss_weight=2.0),  # 分类损失权重
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),  # 边界框回归L1损失（稳健性好）
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),  # IoU损失（未启用）
    ),
    train_cfg=dict(fusion=dict(
        grid_size=[512, 512, 1],  # 网格大小
        voxel_size=voxel_size,  # 体素尺寸
        point_cloud_range=point_cloud_range,  # 点云范围
        out_size_factor=4,  # 输出尺寸缩放因子
        assigner=dict(
            type='HungarianAssigner3D',  # 匈牙利分配器（用于目标匹配）
            cls_cost=dict(type='FocalLossCost', weight=2.0),  # 分类损失代价
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),  # 边界框回归代价
            iou_cost=dict(type='IoUCost', weight=0.0),  # IoU代价（未启用）
            pc_range=point_cloud_range),  # 点云范围
    )),
)

# data pipeline
file_client_args = dict(backend='disk')
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)  # we use nuimages pretrain for 2D detector
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
ida_aug_conf = {
    "resize_lim": (1.0, 1.0),
    "final_dim": (640, 1600),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

dataset_type = 'CustomNuScenesDataset'
data_root = './data/nuscenes/'

train_pipeline = [
    dict(
        type='LoadPointsFromFile',  # 从文件加载点云
        coord_type='LIDAR',  # 坐标类型：激光雷达
        load_dim=5,  # 加载维度
        use_dim=5,  # 使用维度
        file_client_args=file_client_args),  # 文件客户端参数
    dict(
        type='LoadPointsFromMultiSweeps',  # 从多帧加载点云
        sweeps_num=9,  # 加载的帧数
        load_dim=5,  # 加载维度
        use_dim=[0, 1, 2, 3, 4],  # 使用维度
        pad_empty_sweeps=True,  # 填充空帧
        remove_close=True,  # 移除近距离点
        file_client_args=file_client_args),  # 文件客户端参数
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),  # 从文件加载多视图图像
    dict(type='LoadAnnotations3D',  # 加载3D标注
         with_bbox_3d=True,  # 包含3D边界框
         with_label_3d=True,  # 包含3D标签
         with_bbox=True,  # 包含2D边界框
         with_label=True,  # 包含2D标签
         with_bbox_depth=True),  # 包含深度信息
    dict(type='ResizeCropFlipRotImage',  # 图像增强：调整大小、裁剪、翻转、旋转
         data_aug_conf=ida_aug_conf,  # 数据增强配置
         training=True),  # 训练模式
    dict(type='BEVGlobalRotScaleTrans',  # 全局旋转、缩放、平移
         rot_range=[-1.57075, 1.57075],  # 旋转范围
         translation_std=[0, 0, 0],  # 平移标准差
         scale_ratio_range=[0.95, 1.05],  # 缩放比例范围
         reverse_angle=True,  # 反向角度
         training=True),  # 训练模式
    dict(type='BEVRandomFlip3D'),  # 随机翻转3D点云
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),  # 过滤点云范围外的目标
    dict(type='ObjectNameFilter', classes=class_names),  # 过滤不在类别列表中的目标
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),  # 过滤点云范围外的点
    dict(type='NormalizePoints'),  # 点云归一化
    dict(type='PointShuffle'),  # 随机打乱点云
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),  # 多视图图像归一化
    dict(type='PadMultiViewImage', size_divisor=32),  # 多视图图像填充
    dict(type='PETRFormatBundle3D',  # 格式化3D数据
         class_names=class_names,  # 类别名称
         collect_keys=collect_keys + ['prev_exists']),  # 收集的键
    dict(type='Collect3D',  # 收集3D数据
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths',
               'prev_exists'] + collect_keys,  # 收集的键
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d',
                    'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d')),  # 元数据键
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',  # 从文件加载点云
        coord_type='LIDAR',  # 坐标类型：激光雷达
        load_dim=5,  # 加载维度
        use_dim=5,  # 使用维度
        file_client_args=file_client_args),  # 文件客户端参数
    dict(
        type='LoadPointsFromMultiSweeps',  # 从多帧加载点云
        sweeps_num=9,  # 加载的帧数
        load_dim=5,  # 加载维度
        use_dim=[0, 1, 2, 3, 4],  # 使用维度
        pad_empty_sweeps=True,  # 填充空帧
        remove_close=True,  # 移除近距离点
        file_client_args=file_client_args),  # 文件客户端参数
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),  # 从文件加载多视图图像
    dict(type='ResizeCropFlipRotImage',  # 图像增强：调整大小、裁剪、翻转、旋转
         data_aug_conf=ida_aug_conf,  # 数据增强配置
         training=False),  # 测试模式
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),  # 多视图图像归一化
    dict(type='PadMultiViewImage', size_divisor=32),  # 多视图图像填充
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),  # 过滤点云范围外的点
    dict(type='NormalizePoints'),  # 点云归一化
    dict(
        type='MultiScaleFlipAug3D',  # 多尺度翻转增强
        img_scale=(1333, 800),  # 图像尺寸
        pts_scale_ratio=1,  # 点云缩放比例
        flip=False,  # 不翻转
        transforms=[
            dict(
                type='PETRFormatBundle3D',  # 格式化3D数据
                collect_keys=collect_keys + ['prev_exists'],  # 收集的键
                class_names=class_names,  # 类别名称
                with_label=False),  # 不包含标签
            dict(type='Collect3D',  # 收集3D数据
                 keys=['points', 'img', 'prev_exists'] + collect_keys,  # 收集的键
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d',
                            'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'))  # 元数据键
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes2d_temporal_infos_train.pkl',
        pipeline=train_pipeline,
        num_frame_losses=num_frame_losses,
        seq_split_num=2,  # streaming video training
        seq_mode=True,  # streaming video training
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        collect_keys=collect_keys + ['img', 'img_metas'],
        queue_length=queue_length,
        ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl',
        classes=class_names,
        modality=input_modality),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        collect_keys=collect_keys + ['img', 'img_metas'],
        queue_length=queue_length,
        ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl',
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'pts_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(
    type='Fp16OptimizerHook', loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

custom_hooks = []

evaluation = dict(interval=3 * num_iters_per_epoch, pipeline=test_pipeline)
checkpoint_config = dict(interval=1 * num_iters_per_epoch)
find_unused_parameters = False  #### when use checkpoint, find_unused_parameters must be False
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from = None
resume_from = None

# Evaluating bboxes of pts_bbox
# mAP: 0.7304
# mATE: 0.2808
# mASE: 0.2447
# mAOE: 0.2502
# mAVE: 0.2049
# mAAE: 0.1906
# NDS: 0.7481
# Eval time: 92.9s