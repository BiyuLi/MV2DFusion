from matplotlib import cm
import os
import cv2
import torch
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def extract_camera_name(filename):
    """
    使用正则表达式从文件路径中精确提取相机名称

    Args:
        filename: 文件路径字符串

    Returns:
        str: 相机名称，如 'CAM_FRONT', 'CAM_FRONT_LEFT' 等，如果未找到返回None
    """
    camera_patterns = [
        r'CAM_FRONT_LEFT',    # 必须在前，避免被CAM_FRONT匹配
        r'CAM_FRONT_RIGHT',   # 必须在前，避免被CAM_FRONT匹配
        r'CAM_BACK_LEFT',     # 必须在前，避免被CAM_BACK匹配
        r'CAM_BACK_RIGHT',    # 必须在前，避免被CAM_BACK匹配
        r'CAM_FRONT',
        r'CAM_BACK'
    ]
    for pattern in camera_patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(0)
    return None


def extract_frame_token(filename: str):
    """从文件名中提取 frame_token（假设文件名里包含 frame token 部分）"""
    return os.path.basename(filename).split('.')[0]


def merge_img(img_list, rows, cols):
    """把多张图拼接成 (rows x cols) 的大图"""
    h, w, c = img_list[0].shape
    canvas = np.zeros((h * rows, w * cols, c), dtype=np.uint8)
    for idx, img in enumerate(img_list):
        r, c_idx = divmod(idx, cols)
        canvas[r * h:(r + 1) * h, c_idx * w:(c_idx + 1) * w, :] = img
    return canvas


def vis_dets_2d(imgs_meta, imgs_det, score_thresh=0.4, save_dir=None, y_offset=260, return_img=False):
    """
    可视化6个相机的2D检测结果，拼接成2×3的大图
    """
    label_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    label_color_mapping = {
        'car': (255, 255, 0),
        'truck': (255, 255, 0),
        'trailer': (255, 255, 0),
        'bus': (255, 255, 0),
        'construction_vehicle': (255, 255, 0),
        'bicycle': (125, 125, 125),
        'motorcycle': (125, 126, 126),
        'pedestrian': (0, 255, 0),
        'traffic_cone': (255, 255, 255),
        'barrier': (255, 255, 255)
    }

    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT'
    ]
    cam_to_index = {cam: idx for idx, cam in enumerate(cams)}

    def get_cam_index(cam_pos):
        return cam_to_index.get(cam_pos, len(cams))

    frame_token = extract_frame_token(imgs_meta[0].get('filename', 'unknown'))

    sub_plots = []
    for img_meta, dets in zip(imgs_meta, imgs_det):
        filename = img_meta['filename']
        cam_position = extract_camera_name(filename)

        if not os.path.exists(filename):
            print(f"[WARN] Image {filename} not found, skip")
            continue

        img = cv2.imread(filename)
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy()

        for det in dets:
            x1, y1, x2, y2, score, label_idx = det
            if score < score_thresh:
                continue
            # 强制 y 坐标 +260
            x1, y1, x2, y2 = int(x1), int(
                y1) + y_offset, int(x2), int(y2) + y_offset

            if 0 <= label_idx < len(label_names):
                label_name = label_names[int(label_idx)]
                color = label_color_mapping[label_name]

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label_text = f'{label_name}: {score:.2f}'
                (text_width, text_height) = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1
                )[0]
                cv2.rectangle(img, (x1, y1 - text_height),
                              (x1 + text_width, y1), color, -1)
                cv2.putText(img, label_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
            else:
                print(f"[WARN] label_idx {label_idx} not in valid range")

        cv2.putText(img, cam_position, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        sub_plots.append({
            'img': img,
            'pos': cam_position
        })

    # 排序相机
    sorted_sub_plots = sorted(sub_plots, key=lambda x: get_cam_index(x['pos']))
    sorted_sub_plots = [x['img'] for x in sorted_sub_plots]

    if len(sorted_sub_plots) == 6:
        output = merge_img(sorted_sub_plots, 2, 3)  # 拼成2行3列
    else:
        print(
            f"[ERROR] {frame_token}: has only {len(sorted_sub_plots)} inputs")
        return

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"dets_2d_{frame_token}.png")
        cv2.imwrite(save_path, output)
        print(f"大图已保存到: {save_path}")
    if return_img:
        return output


def vis_bev(bboxes, scores, labels, query_idx=None, score_thresh=0.2, return_img=True):
    """
    在 BEV 视角下可视化所有 3D 框，并高亮某个指定 query 的框
    Args:
        bboxes: LiDARInstance3DBoxes
        scores: Tensor [N]
        labels: Tensor [N]
        query_idx: int 或 None, 如果指定则高亮这个框
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # 获取 numpy
    corners = bboxes.corners.cpu().numpy()  # (N, 8, 3)
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_title("BEV Visualization")

    for i in range(len(corners)):
        if scores[i] < score_thresh:
            continue
        # 取 bottom face (x,y)
        box = corners[i, [0, 1, 2, 3], :2]
        poly = plt.Polygon(box, closed=True,
                           edgecolor="blue" if i != query_idx else "red",
                           facecolor="none",
                           linewidth=2 if i != query_idx else 3,
                           linestyle="--" if i != query_idx else "-")
        ax.add_patch(poly)
        if i == query_idx:
            ax.text(box[0, 0], box[0, 1],
                    f"Lidar {i}", color="red", fontsize=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # 转成 numpy 图像
    fig.canvas.draw()
    bev_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    bev_img = bev_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    if return_img:
        return bev_img
    else:
        return None

def get_2d_boxes_from_indices(dets, img_index):
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
def map_global_img_index(global_idx, dets):
    """
    输入一个 global_idx，返回 (cam_id, local_idx)
    """
    count = 0
    for cam_id, cam_dets in enumerate(dets):
        num = len(cam_dets)
        if global_idx < count + num:
            local_idx = global_idx - count
            return cam_id, local_idx
        count += num
    raise IndexError(f"global_idx={global_idx} 超出范围，总检测数={count}")
def map_global_img_indices(global_indices, dets):
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


def vis_bev_segmentation_style(voxel_xyz,
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
            cv2.rectangle(bev_img, (cx - 5, cy - 5),
                          (cx + 5, cy + 5), (0, 0, 255), 2)
        else:
            cv2.rectangle(bev_img, (cx - 3, cy - 3),
                          (cx + 3, cy + 3), (0, 128, 255), 1)
    return bev_img


# def vis_attention(fusion_outs, out_boxes, pts_out_dict, imgs_out_dict,save_dir):
#     # {lidar_idx: [img_query_indices]}
#     lidar2img_dict = fusion_outs['lidar2img_dict']
#     bevmask = fusion_outs['tgt_query_mask']
#     img_metas_det = imgs_out_dict['img_metas_det']
#     dets = imgs_out_dict['dets']
    
#     # 先获取 3D 检测结果
#     bboxes, scores, labels = out_boxes[0]  # batch=0
#     os.makedirs(save_dir, exist_ok=True)
#     # 遍历 lidar2img_dict
#     for lidar_idx, img_indices in lidar2img_dict.items():
#         # 左边 BEV 图
#         bev_img = vis_bev_segmentation_style(
#             voxel_xyz=pts_out_dict['voxel_xyz'],
#             query_pred=pts_out_dict['query_xyz'],
#             query_cat=pts_out_dict['query_cat'],
#             bboxes=bboxes,
#             scores=scores,
#             labels=labels,
#             highlight_idx=lidar_idx,
#             mask=bevmask,
#             bev_range=[-50, 50, -50, 50],
#             out_size=(512, 512)
#         )
#         # 构造每个相机的选中框
#         filtered_dets = []
#         for cam_id, cam_dets in enumerate(dets):
#             # 找出属于该相机的 topk 图像query框
#             selected = [local_idx for (cid, local_idx) in map_global_img_indices(img_indices, dets)
#                         if cid == cam_id]
#             if len(selected) > 0:
#                 selected_tensor = torch.tensor(
#                     selected, device=cam_dets.device, dtype=torch.long)
#                 cam_dets_selected = cam_dets[selected_tensor]
#             else:
#                 cam_dets_selected = cam_dets.new_zeros((0, 6))
#             filtered_dets.append(cam_dets_selected)

#         # 右边 相机拼图
#         cam_img = vis_dets_2d(
#             img_metas_det, filtered_dets, 0.0, return_img=True)

#         # =========================
#         # 调整 BEV 和相机图比例
#         bev_h, bev_w, _ = bev_img.shape
#         cam_h, cam_w, _ = cam_img.shape

#         # 统一高度
#         target_h = max(bev_h, cam_h)

#         # BEV 缩放
#         scale_bev = target_h / bev_h
#         bev_w_new = int(bev_w * scale_bev)
#         bev_img_resized = cv2.resize(bev_img, (bev_w_new, target_h))

#         # 相机图缩放
#         scale_cam = target_h / cam_h
#         cam_w_new = int(cam_w * scale_cam)
#         cam_img_resized = cv2.resize(cam_img, (cam_w_new, target_h))

#         # 拼接
#         canvas = np.ones((target_h, bev_w_new + cam_w_new, 3),
#                          dtype=np.uint8) * 255
#         canvas[:, :bev_w_new] = bev_img_resized
#         canvas[:, bev_w_new:bev_w_new + cam_w_new] = cam_img_resized

#         # 保存
#         out_path = os.path.join(save_dir, f"lidar_{lidar_idx}.jpg")
#         cv2.imwrite(out_path, canvas)
#         print(f"[INFO] 保存到 {out_path}")
# dets 是一个 list，每个相机一个 tensor: [N, 6] (x1, y1, x2, y2, score, class_id)

def get_cat_from_dets(dets, global_idx):
    # dets 是 [cam0_dets, cam1_dets, ...]
    # global_idx 需要先映射成 (cam_id, local_idx)
    cam_id, local_idx = map_global_img_index(global_idx, dets)  

    if local_idx < len(dets[cam_id]):
        return cam_id,int(dets[cam_id][local_idx, -1].item())  # class_id 在最后一列
    else:
        print(f"[WARN] local_idx={local_idx} 超出相机 {cam_id} 的检测数 {len(dets[cam_id])}")
        return -1
def vis_attention(fusion_outs, out_boxes, pts_out_dict, imgs_out_dict,save_dir):
    # {lidar_idx: [img_query_indices]}
    lidar2img_dict = fusion_outs['lidar2img_dict']
    lidar2img_points_dict=fusion_outs['lidar2img_points_dict']
    bevmask = fusion_outs['tgt_query_mask']
    img_metas_det = imgs_out_dict['img_metas_det']
    dets = imgs_out_dict['dets']
    
    # 先获取 3D 检测结果
    bboxes, scores, labels = out_boxes[0]  # batch=0
    os.makedirs(save_dir, exist_ok=True)
    # 遍历 lidar2img_dict
    for lidar_idx, img_indices in lidar2img_dict.items():
        top1, top2 = img_indices[:2]
        # === 获取对应的 points ===
        points = lidar2img_points_dict[lidar_idx]  # [TopN, 3]
        pt1, pt2 = points[0], points[1]
        # === 获取相机ID和对应的类别（从 query_cat） ===
        cid1,cat1 = get_cat_from_dets(dets, top1)
        cid2,cat2 = get_cat_from_dets(dets, top2)
        # === 检查过滤条件 ===
        keep_top2 = True
        if cid1 == cid2:  # 同一个相机
            keep_top2 = False
        elif torch.norm(pt1 - pt2).item() > 0.06:  # 点太远
            keep_top2 = False
        elif cat1 != cat2:  # 类别不同
            keep_top2 = False

        # === 构造最终的候选 indices ===
        filtered_indices = [top1]
        if keep_top2:
            filtered_indices.append(top2)

        # 左边 BEV 图
        bev_img = vis_bev_segmentation_style(
            voxel_xyz=pts_out_dict['voxel_xyz'],
            query_pred=pts_out_dict['query_xyz'],
            query_cat=pts_out_dict['query_cat'],
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            highlight_idx=lidar_idx,
            mask=bevmask,
            bev_range=[-50, 50, -50, 50],
            out_size=(512, 512)
        )
        # 构造每个相机的选中框
        filtered_dets = []
        for cam_id, cam_dets in enumerate(dets):
            # 找出属于该相机的 topk 图像query框
            selected = [local_idx for (cid, local_idx) in map_global_img_indices(filtered_indices, dets)
                        if cid == cam_id]
            if len(selected) > 0:
                selected_tensor = torch.tensor(
                    selected, device=cam_dets.device, dtype=torch.long)
                cam_dets_selected = cam_dets[selected_tensor]
            else:
                cam_dets_selected = cam_dets.new_zeros((0, 6))
            filtered_dets.append(cam_dets_selected)

        # 右边 相机拼图
        cam_img = vis_dets_2d(
            img_metas_det, filtered_dets, 0.0, return_img=True)

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
        canvas = np.ones((target_h, bev_w_new + cam_w_new, 3),
                         dtype=np.uint8) * 255
        canvas[:, :bev_w_new] = bev_img_resized
        canvas[:, bev_w_new:bev_w_new + cam_w_new] = cam_img_resized

        # 保存
        out_path = os.path.join(save_dir, f"lidar_{lidar_idx}.jpg")
        cv2.imwrite(out_path, canvas)
        print(f"[INFO] 保存到 {out_path}")
    # =========================
    # img_index=fusion_outs['topk_img_indices']
    # selected_boxes, mappings = get_2d_boxes_from_indices(dets, img_index)
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
