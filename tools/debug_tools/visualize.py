import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
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

def extract_frame_token(file_path):
    """
    使用正则表达式提取信息
    
    Args:
        file_path: str, 文件路径
        
    Returns:
        str:  最后一组连续数字，如果未找到则返回'unknown'
    """
    # 匹配模式：连续数字
    numbers = re.findall(r'\d+', file_path)
    if numbers:
        return numbers[-1]  # 返回最后一组数字
    return 'unknown'

def merge_img(img_list, row, col):
    if not img_list:
        return
    merged_img = np.zeros((row * img_list[0].shape[0],
                            col * img_list[0].shape[1], 3),
                            dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            index = i * col + j
            if index >= len(img_list):
                break
            image = img_list[index]
            h, w = image.shape[:2]
            merged_img[i * h:(i + 1) * h, j * w:(j + 1) * w] = image
    return merged_img

def vis_dets_2d(imgs_meta, imgs_det, score_thresh = 0.4, save_dir=None):
    """
    可视化6个相机的2D检测结果
    
    Args:
        imgs_meta: list[dict], 长度为6的列表，每个字典包含图像元数据
        imgs_det: list[tensor], 长度为6的列表，每个tensor形状为(n_bbox2d, 6)，包含(x1, y1, x2, y2, score, label)
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
    def get_cam_index(cam_dict):
        cam_pos = cam_dict.get('pos')
        return cam_to_index.get(cam_pos, len(cams))
    frame_token = extract_frame_token(imgs_meta[0].get('filename', 'unknown'))
    sub_plots =[]
    for _, (img_meta, dets) in enumerate(zip(imgs_meta, imgs_det)):
        filename = img_meta['filename']
        cam_position = extract_camera_name(filename)
        if os.path.exists(filename):
            img = cv2.imread(filename)
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy()
        for det in dets:
            x1, y1, x2, y2, score, label_idx = det
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = int(x1), int(y1)+260, int(x2), int(y2)+260
            if 0 <= label_idx < len(label_names):
                label_name = label_names[int(label_idx)]
                color = label_color_mapping[label_name]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label_text = f'{label_name}: {score:.2f}'
                (text_width, text_height) = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                                        thickness=1)[0]
                cv2.rectangle(img, (x1, y1 - text_height),
                                (x1 + text_width, y1), color, -1)
                cv2.putText(img, label_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
            else:
                print(f"{label_idx} not in [0, {len(label_names)}]")
    
        cv2.putText(img, cam_position , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 255, 0), 2)
        sub_plots.append({
            'img':img,
            'pos':cam_position})
        sorted_sub_plots = sorted(sub_plots, key=get_cam_index)
        sorted_sub_plots = [x['img'] for x in sorted_sub_plots]
    if len(sorted_sub_plots) == 6:
        output = merge_img(sorted_sub_plots, 2, 3)
    else:
        print(f"[ERROR] {frame_token}: has only {len(sorted_sub_plots)} inputs")
        return
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        saved_name = f"dets_2d_{frame_token}"
        save_path = os.path.join(save_dir, f"{saved_name}.png")
        cv2.imwrite(save_path, output)
        print(f"图像已保存到: {save_path}")