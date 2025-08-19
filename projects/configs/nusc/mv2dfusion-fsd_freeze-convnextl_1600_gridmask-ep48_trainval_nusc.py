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
voxel_size = [0.2, 0.2, 8]
point_cloud_range = [-54.4, -54.4, -5.0, 54.4, 54.4, 3.0]
sparse_shape = [40, 544, 544]
target_sparse_shape = [20, 272, 272]
fsd_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
seg_voxel_size = (0.2, 0.2, 0.2)
virtual_voxel_size=(0.4, 0.4, 0.4) #(1024, 1024, 16)、
#将类别按相似性分组，目的是：简化模型学习难度（相似目标共享部分特征）、差异化设置阈值
group1 = ['car']
group2 = ['truck', 'construction_vehicle']
group3 = ['bus', 'trailer']
group4 = ['barrier']
group5 = ['motorcycle', 'bicycle']
group6 = ['pedestrian', 'traffic_cone']
group_names = [group1, group2, group3, group4, group5, group6]
seg_score_thresh = [0.2, ] * 3 + [0.1, ] * 3  #前 3 组用 0.2，后 3 组用 0.1，因小目标更难检测
group_lens = [len(group1), len(group2), len(group3), len(group4), len(group5), len(group6)]
#将 10 类分为前 5 类和后 5 类，对应tasks中的两个子任务，实现 "分头部检测"（不同头部专注于不同类别）
head_group1 = fsd_class_names[:5]
head_group2 = fsd_class_names[5:]
tasks = [
    dict(class_names=head_group1),
    dict(class_names=head_group2),
]

queue_length = 1
num_frame_losses = 1

roi_size = 7
roi_strides = [4, 8, 16, 32, 64]
model = dict(
    type='MV2DFusion',
    dataset='nuscenes',
    num_frame_head_grads=num_frame_losses,## 参与头部网络梯度计算的帧数
    num_frame_backbone_grads=num_frame_losses,# 参与主干网络梯度计算的帧数
    num_frame_losses=num_frame_losses,# 参与损失计算的帧数（时序信息利用）
    position_level=2, #位置编码的层级（通过编码增强特征的空间位置信息，提升定位精度）。
    use_grid_mask=True,#启用网格掩码（在特征图上生成局部掩码，强制模型关注重要区域，抑制背景噪声）

    loss_weight_3d=0.1,# 3D检测损失的权重
    loss_weight_pts=1.,# 点云相关损失的权重（点级损失更重要）
    gt_mono_loss=True,# 是否使用基于真值的单目损失（通过真值监督单目视图特征学习，缓解单目深度估计模糊问题）

    img_backbone=dict(
        type='ConvNeXt',# 主干网络类型：ConvNeXt（基于卷积的现代网络，类似Transformer的设计思想）
        arch='large',# 网络规模：large（参数更多，适合复杂场景）
        out_indices=[0, 1, 2, 3],# 输出的特征层索引（多尺度特征，满足不同大小目标的检测需求）
        drop_path_rate=0.4,# 随机深度比率（正则化，防止过拟合）
        layer_scale_init_value=0.,# 层缩放初始化值（控制残差连接的权重）
        gap_before_final_norm=False,# 最终归一化前是否使用全局平均池化
        use_grn=True,# 是否使用全局响应归一化（GRN，增强特征判别性）
        with_cp=True,# 是否使用checkpointing（节省内存，适合大模型训练）
    ),
    img_neck=dict(
        type='FPN',# 颈部网络类型：特征金字塔网络（FPN）
        in_channels=[192, 384, 768, 1536],# 输入特征通道（对应ConvNeXt的4个输出层）
        out_channels=256,# 输出特征通道（统一多尺度特征的通道数）
        num_outs=5),# 输出的特征层数量（生成5个不同尺度的特征图）
    img_roi_extractor=dict(
        type='SingleRoIExtractor',# ROI提取器类型：单ROI提取器
        roi_layer=dict(type='RoIAlign', output_size=roi_size, sampling_ratio=-1),# ROI对齐层（精确提取感兴趣区域特征）
        featmap_strides=roi_strides[:-1],# 特征图步长（对应不同尺度特征图的下采样率）
        out_channels=256, ),# 输出特征通道
    # faster rcnn
    img_roi_head=dict(
        type='TwoStageDetectorWrapper',
        rpn_head=dict(
            type='RPNHead',# 候选框生成网络
            in_channels=256,# 输入特征通道（FPN输出）
            feat_channels=256, # 特征通道数
            anchor_generator=dict(# 锚点生成器（预设候选框）
                type='AnchorGenerator',
                scales=[8],# 锚点尺度
                ratios=[0.5, 1.0, 2.0], # 锚点宽高比
                strides=roi_strides),# 不同特征图上的锚点步长
            bbox_coder=dict(# 边界框编码器（将坐标差转换为回归目标）
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],# 均值
                target_stds=[1.0, 1.0, 1.0, 1.0]),# 标准差
            loss_cls=dict( # 分类损失（判断锚点是否为目标）
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),# 边界框回归损失
        ),
        roi_head=dict(
            type='CascadeRoIHead',# 级联RoI头（多阶段细化检测结果）
            num_stages=3,# 级联阶段数（3阶段逐步优化）
            stage_loss_weights=[1, 0.5, 0.25],# 锚点宽高比
            bbox_roi_extractor=dict( # 与img_roi_extractor类似，用于级联阶段的ROI提取
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=roi_strides[:-1]),
            bbox_head=[# 3个级联的边界框头（每个阶段参数逐步精细化）
                dict(
                    type='Shared2FCBBoxHead',# 共享2个全连接层的边界框头
                    in_channels=256,# 输入特征通道
                    fc_out_channels=1024, # 全连接层输出通道
                    roi_feat_size=7,# ROI特征大小（7x7）
                    num_classes=10,# 目标类别数（nuScenes数据集有10类目标）
                    bbox_coder=dict(# 边界框编码器（目标标准差逐渐减小，精细化回归）
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]), # 第一阶段：较大标准差
                    reg_class_agnostic=True, # 回归与类别无关（所有类别共享回归参数）
                    loss_cls=dict(# 分类损失
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    # loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                    #                loss_weight=1.0),
                    reg_decoded_bbox=True,# 基于解码后的边界框计算损失（更直观）
                    loss_bbox=dict(type='GIoULoss', loss_weight=10.0)# 边界框损失（GIoU更鲁棒）
                ),
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=10,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.05, 0.05, 0.1, 0.1]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    reg_decoded_bbox=True,
                    loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
                ),
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=10,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.033, 0.033, 0.067, 0.067]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    # loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                    #                loss_weight=1.0),
                    reg_decoded_bbox=True,
                    loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
                )
            ],
            mask_head=None,
        ),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(# 锚点与GT的匹配规则
                    type='MaxIoUAssigner',# 基于IoU的最大匹配策略
                    pos_iou_thr=0.7,# IoU≥0.7的锚点视为正样本（目标）
                    neg_iou_thr=0.3,# IoU≤0.3的锚点视为负样本（背景）
                    min_pos_iou=0.3,# 正样本的最小IoU阈值（低于此则不匹配）
                    match_low_quality=True,# 允许低质量匹配（如GT未匹配到高IoU锚点时，匹配最优锚点）
                    ignore_iof_thr=-1),# 忽略与GT的IoF（交叠率）阈值（-1表示不忽略）
                sampler=dict(# 样本采样策略
                    type='RandomSampler',
                    num=256, # 每批采样256个样本
                    pos_fraction=0.5,# 正样本占比50%（128正+128负）
                    neg_pos_ub=-1,# 负样本相对正样本的最大比例（-1表示无限制）
                    add_gt_as_proposals=False),# 不将GT直接作为候选框
                allowed_border=0, # 允许候选框超出图像边界的范围（0表示严格在图像内）
                pos_weight=-1,# 正样本的损失权重（-1表示自动平衡）
                debug=False),# 是否输出调试信息
            rpn_proposal=dict(#去除高度重叠的候选框，保留有潜力的目标区域，为后续 RCNN 阶段减负
                nms_pre=2000,# NMS前保留的候选框数量（减少计算量）
                nms_post=2000,# NMS后保留的候选框数量
                max_per_img=2000, # 单张图像最终保留的候选框上限
                nms=dict(type='nms', iou_threshold=0.7),# NMS算法（IoU≥0.7的候选框视为重复，保留分数最高的）
                min_bbox_size=0),# 候选框的最小尺寸（0表示不限制）
            rcnn=[#对应前文的CascadeRoIHead，3 个阶段的配置逐步提高正样本 IoU 阈值，实现 "从粗到细" 的检测优化：
                dict(# 第一阶段：宽松匹配
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,# 正样本IoU≥0.5（较宽松）
                        neg_iou_thr=0.5,# 负样本IoU≤0.5
                        min_pos_iou=0.5,
                        match_low_quality=False,# 级联阶段不允许低质量匹配
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,# 每批采样512个样本
                        pos_fraction=0.25,# 正样本占比25%（128正+384负）
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True), # 将GT加入候选框（强化正样本学习）
                    mask_size=28,
                    pos_weight=-1,
                    debug=False),
                dict(# 第二阶段：中等严格
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.6,
                        min_pos_iou=0.6,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False),
                dict( # 第三阶段：严格匹配
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.7,
                        min_pos_iou=0.7,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False)
            ]),
        test_cfg=dict(#推理时通过低分数阈值保留更多候选框，再通过 NMS 去除重复，最终输出少量高质量检测结果。
            min_bbox_size=4,# 过滤尺寸小于4的候选框（去除噪声）
            rpn=dict(# RPN推理配置
                nms_pre=1000, # 推理时NMS前保留1000个候选框（减少计算）
                max_per_img=1000,# 单图RPN输出上限
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(# RCNN推理配置
                score_thr=0.05,# 过滤分数<0.05的框（保留更多潜在目标）
                nms=dict(type='nms', iou_threshold=0.6, class_agnostic=True),# 跨类别NMS（IoU≥0.6视为重复）
                max_per_img=60,)),# 单图最终输出60个检测结果（控制冗余）
    ),
    img_query_generator=dict(#将图像特征转化为结构化查询向量（包含类别、尺寸、深度分布等），为融合头提供图像端的先验信息。
        type='ImageDistributionQueryGenerator',# 基于图像分布的查询生成器
        prob_bin=50,# 概率分箱数（将连续概率离散化为50个区间）
        depth_range=[0.1, 90],# 深度范围（图像中目标可能的距离）
        gt_guided=False,# 不使用GT引导查询生成（纯模型预测）
        gt_guided_loss=1.,
        with_cls=True,
        with_size=True,# 生成包含类别和尺寸信息的查询
        # 特征处理网络
        with_avg_pool=True,
        # 1个共享卷积+1个共享全连接层
        num_shared_convs=1,
        num_shared_fcs=1,
        in_channels=256,# 输入特征通道（来自FPN）
        fc_out_channels=1024,
        roi_feat_size=roi_size,
        extra_encoding=dict(# 额外编码（如相机内参）
            num_layers=2,
            feat_channels=[512, 256],
            features=[dict(type='intrinsic', in_channels=16, )]# 处理16维相机内参特征
        ),
    ),
    # 基于 FSDv2（Fully Sparse Detection v2）架构，处理点云数据并生成 3D 检测特征，是点云端的核心特征提取模块。
    pts_backbone=dict(
        type='SingleStageFSDV2',# 单阶段全稀疏检测架构
        freeze=True,# 冻结部分参数（可能用于预训练权重固定）
        norm_eval=True,# 推理模式下冻结BN层
        segmentor=dict(# 点云分割器（生成点级语义特征）
            type='VoteSegmentor', # 带投票机制的点云分割器
            tanh_dims=[],# 不使用tanh激活的维度（无特殊约束）
            # 1. 体素化层：将点云划分为体素
            voxel_layer=dict(
                voxel_size=seg_voxel_size,
                max_num_points=-1,# 每个体素最大点数（-1表示无限制）
                point_cloud_range=point_cloud_range,
                max_voxels=(-1, -1)
            ),
            # 2. 体素编码器：将体素内点云编码为体素特征
            voxel_encoder=dict(
                type='DynamicScatterVFE',# 动态聚集体素编码器（自适应聚集点特征）
                in_channels=5, # 输入点云维度（x,y,z,intensity,时间戳或其他特征）
                feat_channels=[64, 64], # 两层卷积通道数（逐步升维）
                voxel_size=seg_voxel_size, # 体素尺寸（与voxel_layer一致）
                with_cluster_center=True,# 编码体素内点的聚类中心（增强空间信息）
                with_voxel_center=True,# 编码体素中心坐标（相对位置信息）
                point_cloud_range=point_cloud_range,
                norm_cfg=dict(type='MyBN1d', eps=1e-3, momentum=0.01),# 自定义1D BatchNorm（稳定训练）
                unique_once=True,# 避免重复计算体素特征
            ),
             # 3. 中间编码器：连接体素编码器与主干网络
            middle_encoder=dict(
                type='PseudoMiddleEncoderForSpconvFSD',# 适配稀疏卷积的中间层
            ),
            # 4. 稀疏主干网络：提取多尺度体素特征
            backbone=dict(
                type='SimpleSparseUNet',# 稀疏U-Net（编码-解码结构，保留多尺度特征）
                in_channels=64, # 输入特征通道（来自voxel_encoder输出）
                sparse_shape=sparse_shape,# 稀疏体素网格维度[40,544,544]（全局配置）
                order=('conv', 'norm', 'act'),# 卷积→归一化→激活的操作顺序
                norm_cfg=dict(type='MyBN1d', eps=1e-3, momentum=0.01),
                base_channels=64,
                output_channels=128, # dummy
                encoder_channels=((128, ), (128, 128, ), (128, 128, ), (128, 128, 128), (256, 256, 256), (256, 256, 256)),# 编码器各层通道数（逐步升维，下采样）
                encoder_paddings=((1, ), (1, 1, ), (1, 1, ), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
                decoder_channels=((256, 256, 256), (256, 256, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128)),# 解码器各层通道数（逐步降维，上采样）
                decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1), (1, 1)), # decoder paddings seem useless in SubMConv
                return_multiscale_features=True, # 输出多尺度特征（用于后续融合）
            ),
            # 5. 解码颈部：将体素特征映射回点云特征
            decode_neck=dict(
                type='Voxel2PointScatterNeck',# 体素到点云的散射映射
                voxel_size=seg_voxel_size,
                point_cloud_range=point_cloud_range,# 通过体素坐标反推点云特征
            ),
             # 6. 分割头：输出点云语义分割结果+投票损失
            segmentation_head=dict(
                type='VoteSegHead',# 带投票损失的分割头
                in_channel=67 + 64, # 输入通道（67维点特征+64维体素特征）
                hidden_dims=[128, 128],# 两层隐藏层（特征细化）
                num_classes=len(class_names),# 目标类别数（10类）
                dropout_ratio=0.0,
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type='MyBN1d'),
                act_cfg=dict(type='ReLU'),
                # 语义分割损失（分类损失）
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    class_weight=[1.0, ] * len(class_names) + [0.1,],# 背景类权重降低（减少背景干扰）
                    loss_weight=10.0), # 分割损失权重（强调语义学习）
                loss_vote=dict(# 投票损失（核心创新：引导点向目标中心聚集）
                    type='L1Loss',
                    loss_weight=1.0),
            ),
            train_cfg=dict(
                point_loss=True,
                score_thresh=seg_score_thresh, # for training log
                class_names=fsd_class_names, # for training log
                group_names=group_names,
                group_lens=group_lens,
            ),
        ),
        #真实点云在远距离或遮挡区域通常稀疏，通过生成 “虚拟点云” 补充这些区域的特征，提升检测鲁棒性。
        #基于现有点云特征，通过 MLP 生成虚拟点的特征和位置，填补稀疏区域；
        #“恢复特征” 模块进一步优化虚拟点与真实点的特征一致性，避免引入噪声。
        virtual_point_projector=dict(
            in_channels=83 + 64,# 输入特征通道（83维原始特征+64维编码特征）
            hidden_dims=[64, 64],# 两层MLP（特征降维与增强）
            norm_cfg=dict(type='MyBN1d'),# 归一化稳定训练

            ori_in_channels=67 + 64,# 原始点特征输入通道
            ori_hidden_dims=[64, 64],# 原始特征处理MLP

            # TODO: optional
            recover_in_channels=128 + 3, # with point2voxel offset# 恢复特征输入（128维特征+3维偏移）
            recover_hidden_dims=[128, 128],# 恢复特征MLP（精细化虚拟点特征）
        ),
        #融合不同尺度的点云特征，平衡大目标（需大尺度特征）和小目标（需小尺度特征）的检测需求。
        multiscale_cfg=dict(
            multiscale_levels=[0, 1, 2],# 融合第0/1/2层多尺度特征
            projector_hiddens=[[256, 128], [128, 128], [128, 128]],# 各层特征投影器（降维融合）
            fusion_mode='avg', # 融合方式为平均（简单有效，避免过拟合）
            target_sparse_shape=target_sparse_shape,# 目标稀疏形状[20,272,272]（融合后特征尺寸）
            norm_cfg=dict(type='MyBN1d'),
        ),
        # 虚拟体素编码器（处理虚拟点生成的体素）
        voxel_encoder=dict(
            type='DynamicScatterVFE', # 复用动态聚集编码器
            in_channels=67, # 输入通道（虚拟点特征维度）
            feat_channels=[64, 128],# 特征升维（64→128）
            voxel_size=virtual_voxel_size,# 虚拟体素尺寸(0.4,0.4,0.4)（比真实体素大，降低计算量）
            with_cluster_center=True,
            with_voxel_center=True,
            point_cloud_range=point_cloud_range,
            norm_cfg=dict(type='MyBN1d', eps=1e-3, momentum=0.01),
            unique_once=True,
        ),
        # 虚拟体素主干网络
        backbone=dict(
            type='VirtualVoxelMixer',# 虚拟体素混合网络（融合真实与虚拟体素特征）
            in_channels=128,# 输入通道（来自虚拟体素编码器）
            sparse_shape=target_sparse_shape,# 目标稀疏形状[20,272,272]
            order=('conv', 'norm', 'act'),
            norm_cfg=dict(type='MyBN1d', eps=1e-3, momentum=0.01),
            base_channels=64,
            output_channels=128,
            encoder_channels=((64,), (64, 64,), (64, 64,),),# 编码器通道（逐步提取高层特征）
            encoder_paddings=((1,), (1, 1,), (1, 1,),),
            decoder_channels=((64, 64, 64), (64, 64, 64), (64, 64, 64)),# 解码器通道（恢复细节特征）
            decoder_paddings=((1, 1), (1, 1), (1, 1),),  # decoder paddings seem useless in SubMConv
        ),
        #5. 点云检测头（bbox_head: FSDV2Head）
        bbox_head=dict(
            type='FSDV2Head',
            as_rpn=False,
            num_classes=len(class_names),
            bbox_coder=dict(type='BasePointBBoxCoder', code_size=10),# 10维边界框编码（3D位置+3D尺寸+旋转+速度）
            loss_cls=dict(# 损失函数（分类+回归）
                type='FocalLoss',#  focal损失解决类别不平衡
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=4.0),
            loss_center=dict(type='L1Loss', loss_weight=0.5), # 中心坐标回归损失
            loss_size=dict(type='L1Loss', loss_weight=0.5),# 尺寸回归损失
            loss_rot=dict(type='L1Loss', loss_weight=0.2),# 旋转角回归损失
            loss_vel=dict(type='L1Loss', loss_weight=0.2), # 速度回归损失
            # 特征处理
            in_channel=128,# 输入特征通道（来自虚拟体素主干）
            shared_mlp_dims=[256, 256],# 共享MLP（特征细化）
            train_cfg=None,
            test_cfg=None,
            norm_cfg=dict(type='MyBN1d'),
            # 多任务配置
            tasks=tasks,# 分任务检测（前5类/后5类，见前文）
            class_names=fsd_class_names,
            common_attrs=dict(# 目标属性预测配置
                center=(3, 2, 128), dim=(3, 2, 128), rot=(2, 2, 128), vel=(2, 2, 128)
                # (out_dim, num_layers, hidden_dim)
            ),
            num_cls_layer=2,
            cls_hidden_dim=128,
            separate_head=dict(# 分头部处理不同任务（避免任务干扰）
                type='FSDSeparateHead',
                norm_cfg=dict(type='MyBN1d'),
                act='relu',
            ),
        ),
        train_cfg=dict(
            score_thresh=seg_score_thresh,# 分割分数阈值（前3类0.2，后3类0.1）
            sync_reg_avg_factor=True,# 基于目标质心分配样本（更精准匹配）
            batched_group_sample=True,# 按类别组批量采样（平衡各类样本）
            offset_weight='max',
            class_names=fsd_class_names,
            group_names=[group1, group2, group3, group4, group5, group6],
            centroid_assign=True,
            disable_pretrain=True,
            disable_pretrain_topks=[500, ] * len(class_names),
        ),
        test_cfg=dict(
            score_thresh=seg_score_thresh,
            batched_group_sample=True,
            offset_weight='max',
            class_names=fsd_class_names,
            group_names=[group1, group2, group3, group4, group5, group6],
            use_rotate_nms=True,# 旋转NMS（适应3D目标方向差异）
            nms_pre=-1,
            nms_thr=0.25,# NMS IoU阈值（过滤重复框）
            # score_thr=0.1,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500,# 单类最大输出500个框
            all_task_max_num=200,# 全任务最大输出200个框（控制冗余）
        ),
    ),
    #从点云特征中生成 “查询向量”，为后续多模态融合提供点云端的语义和空间信息。
    pts_query_generator=dict(
        type='PointCloudQueryGenerator',
        in_channels=128,# 输入特征通道（来自点云主干）
        hidden_channel=128,# 隐藏层通道（特征降维）
        pts_use_cat=False,# 不拼接多尺度特征（避免维度膨胀）
    ),
    # fusion head
    #融合图像特征（来自图像主干）和点云特征（来自点云主干），输出最终 3D 检测结果，是多模态协同的核心。
    fusion_bbox_head=dict(
        type='MV2DFusionHead',
        prob_bin=50,# 概率分箱数（将连续概率分布离散化为50个区间，用于特征编码）
        post_bev_nms_ops=[0], # BEV（鸟瞰图）后处理的NMS操作索引（[0]表示仅执行一次NMS）
        num_classes=10,# 10类目标
        in_channels=256,# 输入特征通道数（图像与点云特征融合后的通道维度）
        num_query=300,# 生成300个查询向量（潜在目标）
        memory_len=12 * 256,# 特征记忆长度（用于时序特征缓存，12×256=3072，适配多帧特征）
        topk_proposals=256,# 从候选框中筛选Top-256个高置信度框用于融合
        num_propagated=256,# 传播到下一层的特征数量（保持特征传播效率）
        with_ego_pos=True,# 融合时引入ego车辆位置信息（补偿车辆自身运动对检测的影响）
        match_with_velo=True, # 匹配时考虑目标速度信息（提升动态目标检测精度）
        scalar=10,  ##noise groups# 噪声分组系数（控制训练时添加的噪声强度，增强模型泛化性）
        noise_scale=1.0,# 噪声缩放因子（调节噪声幅度）
        dn_weight=1.0,  ## 去噪训练（DN，Denoising Training）的损失权重（增强查询向量抗干扰能力）
        split=0.75,  ## 正样本比例（训练时正样本占比75%，平衡正负样本）
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],# 3D边界框各维度的回归权重
        transformer=dict(
            type='MV2DFusionTransformer',
            decoder=dict(
                type='MV2DFusionTransformerDecoder',# 融合解码器
                return_intermediate=True,# 返回中间层结果（用于辅助训练或集成预测）
                num_layers=6,# 解码器层数（6层逐步优化特征，平衡精度与效率）
                transformerlayers=dict(# 每层解码器的结构
                    type='MV2DFusionTransformerDecoderLayer',
                    attn_cfgs=[# 注意力配置（自注意力+交叉注意力）
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MixedCrossAttention',
                            embed_dims=256,
                            num_groups=8, # 特征分组数（与注意力头数匹配）
                            num_levels=4,# 融合的特征层级（多尺度特征融合）
                            num_cams=6,# 相机数量（nuScenes数据集通常用6个相机）
                            dropout=0.1,
                            num_pts=13,# 点云采样点数（每个目标采样13个点特征）
                            bias=2.,# 注意力偏置（增强重要特征权重）
                            attn_cfg=dict(
                                type='PETRMultiheadFlashAttention',# 高效Flash注意力（加速计算）
                                batch_first=False,
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),),
                    ],
                    feedforward_channels=2048,# 前馈网络通道数（特征映射能力，2048=256×8）
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=(# 操作顺序：自注意力→归一化→交叉注意力→归一化→前馈网络→归一化
                        'self_attn', 'norm',
                        'cross_attn', 'norm',
                        'ffn', 'norm')
                ),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',# 无NMS编码器（替代传统NMS，避免阈值敏感问题）
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,# 最大输出框数量（与num_query一致，避免冗余）
            voxel_size=voxel_size, # 体素尺寸（用于坐标映射）
            num_classes=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,# 难样本加权系数（增大难样本损失）
            alpha=0.25,# 正负样本权重比（正样本占25%）
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),# 边界框回归L1损失（稳健性好）
        loss_iou=dict(type='GIoULoss', loss_weight=0.0), ),
    train_cfg=dict(fusion=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),  # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range), )))

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
    "resize_lim": (0.75, 1.35),
    "final_dim": (640, 1600),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

dataset_type = 'CustomNuScenesDataset'
data_root = './data/nuscenes/'

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='NormalizePoints'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys + ['prev_exists'],
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D',
                 keys=['points', 'img', 'prev_exists'] + collect_keys ,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d',
                            'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'))
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
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

# TODO: this config is released for inference only
#   the training code has not been completely checked yet

# Accumulating metric data
# Calculating metrics
# Saving metrics to: /tmp/tmpsejislvg/nuscenes-metrics
# mAP: 0.7448
# mATE: 0.2446
# mASE: 0.2285
# mAOE: 0.2688
# mAVE: 0.1990
# mAAE: 0.1149
# NDS: 0.7668
# Eval time: 283.5s
# Completed evaluation for test phase