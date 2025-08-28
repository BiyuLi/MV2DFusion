# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import cv2
# import os




# def extract_frame_token(file_path):
#     """
#     使用正则表达式提取信息
    
#     Args:
#         file_path: str, 文件路径
        
#     Returns:
#         str:  最后一组连续数字，如果未找到则返回'unknown'
#     """
#     # 匹配模式：连续数字
#     numbers = re.findall(r'\d+', file_path)
#     if numbers:
#         return numbers[-1]  # 返回最后一组数字
#     return 'unknown'

# def merge_img(img_list, row, col):
#     if not img_list:
#         return
#     merged_img = np.zeros((row * img_list[0].shape[0],
#                             col * img_list[0].shape[1], 3),
#                             dtype=np.uint8)
#     for i in range(row):
#         for j in range(col):
#             index = i * col + j
#             if index >= len(img_list):
#                 break
#             image = img_list[index]
#             h, w = image.shape[:2]
#             merged_img[i * h:(i + 1) * h, j * w:(j + 1) * w] = image
#     return merged_img

# def vis_dets_2d(imgs_meta, imgs_det, score_thresh = 0.4, save_dir=None):
#     """
#     可视化6个相机的2D检测结果
    
#     Args:
#         imgs_meta: list[dict], 长度为6的列表，每个字典包含图像元数据
#         imgs_det: list[tensor], 长度为6的列表，每个tensor形状为(n_bbox2d, 6)，包含(x1, y1, x2, y2, score, label)
#     """
#     label_names = [
#         'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
#         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
#     ]
#     label_color_mapping = {
#     'car': (255, 255, 0),
#     'truck': (255, 255, 0),
#     'trailer': (255, 255, 0),
#     'bus': (255, 255, 0),
#     'construction_vehicle': (255, 255, 0),
#     'bicycle': (125, 125, 125),
#     'motorcycle': (125, 126, 126),
#     'pedestrian': (0, 255, 0),
#     'traffic_cone': (255, 255, 255),
#     'barrier': (255, 255, 255)
#     }
#     cams = [
#         'CAM_FRONT_LEFT',
#         'CAM_FRONT',
#         'CAM_FRONT_RIGHT',
#         'CAM_BACK_LEFT',
#         'CAM_BACK',
#         'CAM_BACK_RIGHT'
#     ]
#     cam_to_index = {cam: idx for idx, cam in enumerate(cams)}
#     def get_cam_index(cam_dict):
#         cam_pos = cam_dict.get('pos')
#         return cam_to_index.get(cam_pos, len(cams))
#     frame_token = extract_frame_token(imgs_meta[0].get('filename', 'unknown'))
#     sub_plots =[]
#     for _, (img_meta, dets) in enumerate(zip(imgs_meta, imgs_det)):
#         filename = img_meta['filename']
#         cam_position = extract_camera_name(filename)
#         if os.path.exists(filename):
#             img = cv2.imread(filename)
#         if isinstance(dets, torch.Tensor):
#             dets = dets.cpu().numpy()
#         for det in dets:
#             x1, y1, x2, y2, score, label_idx = det
#             if score < score_thresh:
#                 continue
#             x1, y1, x2, y2 = int(x1), int(y1)+260, int(x2), int(y2)+260
#             if 0 <= label_idx < len(label_names):
#                 label_name = label_names[int(label_idx)]
#                 color = label_color_mapping[label_name]
#                 cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#                 label_text = f'{label_name}: {score:.2f}'
#                 (text_width, text_height) = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
#                                         thickness=1)[0]
#                 cv2.rectangle(img, (x1, y1 - text_height),
#                                 (x1 + text_width, y1), color, -1)
#                 cv2.putText(img, label_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                             (255, 255, 255), 1, cv2.LINE_AA)
#             else:
#                 print(f"{label_idx} not in [0, {len(label_names)}]")
    
#         cv2.putText(img, cam_position , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
#                             (0, 255, 0), 2)
#         sub_plots.append({
#             'img':img,
#             'pos':cam_position})
#         sorted_sub_plots = sorted(sub_plots, key=get_cam_index)
#         sorted_sub_plots = [x['img'] for x in sorted_sub_plots]
#     if len(sorted_sub_plots) == 6:
#         output = merge_img(sorted_sub_plots, 2, 3)
#     else:
#         print(f"[ERROR] {frame_token}: has only {len(sorted_sub_plots)} inputs")
#         return
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#         saved_name = f"dets_2d_{frame_token}"
#         save_path = os.path.join(save_dir, f"{saved_name}.png")
#         cv2.imwrite(save_path, output)
#         print(f"图像已保存到: {save_path}")
import os
import cv2
import torch
import numpy as np
import re
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
            x1, y1, x2, y2 = int(x1), int(y1) + y_offset, int(x2), int(y2) + y_offset

            if 0 <= label_idx < len(label_names):
                label_name = label_names[int(label_idx)]
                color = label_color_mapping[label_name]

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label_text = f'{label_name}: {score:.2f}'
                (text_width, text_height) = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1
                )[0]
                cv2.rectangle(img, (x1, y1 - text_height), (x1 + text_width, y1), color, -1)
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
        print(f"[ERROR] {frame_token}: has only {len(sorted_sub_plots)} inputs")
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
            ax.text(box[0, 0], box[0, 1], f"Lidar {i}", color="red", fontsize=10)

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


