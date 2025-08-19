## Inference

### single-gpu
```bash
# R50
python tools/test.py projects/configs/nusc/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py weights/r50_latest.pth --show-dir work_dirs/vis_nuscmv2dfusion-fsd_freeze-r50_1600_gridmask-ep24 --eval bbox

# ConvNext
python tools/test.py projects/configs/nusc/mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep48_trainval_nusc.py weights/convnext_ep48.pth --show-dir work_dirs/vis_nuscmv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep48 --eval bbox
```

### multi-gpus
```bash
# R50
python tools/dist_train.sh projects/configs/nusc/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py weights/r50_latest.pth --show-dir work_dirs/vis_nuscmv2dfusion-fsd_freeze-r50_1600_gridmask-ep24 --eval bbox

# ConvNext
python tools/dist_train.sh projects/configs/nusc/mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep48_trainval_nusc.py weights/convnext_ep48.pth --show-dir work_dirs/vis_nuscmv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep48 --eval bbox
```

## Train

### ConvNext backbone

### R50 backbone

