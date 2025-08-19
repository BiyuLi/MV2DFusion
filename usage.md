## Inference

### single-gpu
```bash
# R50
python tools/test.py projects/configs/nusc/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py weights/r50_latest.pth --eval bbox

# ConvNext
python tools/test.py projects/configs/nusc/mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep48_trainval_nusc.py weights/convnext_ep48.pth --eval bbox
```

### multi-gpus
```bash
# R50
python tools/dist_test.sh projects/configs/nusc/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py weights/r50_latest.pth --eval bbox

# ConvNext
python tools/dist_test.sh projects/configs/nusc/mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep48_trainval_nusc.py weights/convnext_ep48.pth --eval bbox
```

## Train

### single-gpu
```bash
# R50b
python tools/train.py projects/configs/nusc/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc_debug.py --work-dir work_dirs/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_debug

# ConvNext
python tools/train.py projects/configs/nusc/mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep24_nusc.py --work-dir work_dirs/mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep24
```

### multi-gpus
```bash
# R50b
bash tools/dist_train.sh projects/configs/nusc/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py --work-dir work_dirs/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24 8

# ConvNext
bash tools/dist_train.sh projects/configs/nusc/mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep48_trainval_nusc.py --work-dir work_dirs/mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep48 8
```