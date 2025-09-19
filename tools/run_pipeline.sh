#!/bin/bash
# 用法： ./run_pipeline.sh <nuScenes数据集路径> <检测结果保存路径>
# 例如： ./run_pipeline.sh /data/nuscenes ./result/detections

# 1. 参数检查
if [ $# -lt 2 ]; then
  echo "用法：$0 <path_to_nuscenes> <detections_output_dir>"
  exit 1
fi

NUSC_PATH=$1            # nuScenes 数据集路径
DET_DIR=$2              # MV2D检测结果保存路径
DET_JSON="$DET_DIR/pts_bbox/results_nusc.json"

# 确保检测结果保存目录存在
mkdir -p "$DET_DIR"

# 2. 运行检测模型
# echo "=== 运行 MV2D 检测模型 ==="
# python tools/test.py \
#   projects/configs/nusc/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py \
#   weights/r50_latest.pth \
#   --format-only \
#   --format-save-dir "$DET_DIR"

# 检查检测结果文件是否生成
if [ ! -f "$DET_JSON" ]; then
  echo "错误：未在 $DET_JSON 找到检测结果，请检查检测模型输出路径或文件名！"
  exit 1
fi

# 3. 提取 nuScenes 数据
echo "=== 提取 nuScenes 数据 ==="
bash ImmortalTracker-for-CTRL/preparedata/nuscenes/nu_preparedata.sh "$NUSC_PATH"

# 4. 转换检测结果
echo "=== 转换检测结果 ==="
bash ImmortalTracker-for-CTRL/preparedata/nuscenes/nu_convert_detection.sh "$DET_JSON" cp

# 5. 运行 ImmortalTracker 跟踪
echo "=== 运行 ImmortalTracker 跟踪 ==="
python ImmortalTracker-for-CTRL/main_nuscenes_all_history2.py \
  --name cp_plus \
  --det_name cp \
  --config_path ImmortalTracker-for-CTRL/configs/nu_configs/cp_plus.yaml \
  --process 2

echo "=== 全流程完成 ==="
