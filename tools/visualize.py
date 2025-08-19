import os
import tqdm
import json
from visual_nuscenes import NuScenes
from pathlib import Path
import sys

CUR_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CUR_DIR))

use_gt = False
out_dir = './result_vis'
result_json = "test/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc/Tue_Aug_19_14_42_38_2025/pts_bbox/results_nusc"
dataroot='data/nuscenes'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

nusc_gt = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True, pred = False, annotations = "sample_annotation")
nusc_pred = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.25)

with open('{}.json'.format(result_json)) as f:
    table = json.load(f)
tokens = list(table['results'].keys())

for token in tqdm.tqdm(tokens[::50]):
    nusc_gt.render_sample(token, out_path = os.path.join(out_dir,token+"_gt.png"), verbose=False)
    nusc_pred.render_sample(token, out_path = os.path.join(out_dir,token+"_pred.png"), verbose=False)

