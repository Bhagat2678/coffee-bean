"""
Module: evaluate.py
Description: Evaluate a trained YOLO model on the validation set.
             Computes per-image TP/FP/FN analysis and overall metrics.

Usage:
    python scripts/evaluate.py <model_path> <data_yaml> [conf]
"""

import os
import sys
import yaml
import json
import cv2
from ultralytics import YOLO
import numpy as np

model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/best.pt'
data_yaml = sys.argv[2] if len(sys.argv) > 2 else 'datasets/coffee_beans.yaml'
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01

with open(data_yaml, 'r') as f:
    raw = f.read()
    # replace tabs which may break YAML parsing
    raw = raw.replace('\t', '    ')
    data = yaml.safe_load(raw)

dataset_root = data.get('path') or os.path.dirname(data_yaml)
val_rel = data.get('val')
if not val_rel:
    print('No val set found in', data_yaml)
    sys.exit(1)

val_dir = os.path.join(dataset_root, val_rel)
img_exts = ['.jpg', '.jpeg', '.png']

# collect image files
img_files = []
for fname in os.listdir(val_dir):
    if os.path.splitext(fname)[1].lower() in img_exts:
        img_files.append(os.path.join(val_dir, fname))
img_files.sort()

print(f'Found {len(img_files)} validation images in {val_dir}')

model = YOLO(model_path)
print('Model loaded:', model_path)

# Run ultralytics built-in val for overall metrics
print('\nRunning built-in model.val() for overall metrics...')
val_res = model.val(data=data_yaml, imgsz=640, conf=conf, iou=0.5)
print('Built-in validation complete. Results:')
try:
    print(json.dumps(val_res, indent=2))
except Exception:
    print(val_res)

# Per-image analysis for failure cases (IoU threshold 0.5)
print('\nRunning per-image analysis (TP/FP/FN) @ IoU 0.5...')
IoU_thr = 0.5
failure_cases = []
summary = {'TP':0,'FP':0,'FN':0}

for img_path in img_files:
    # corresponding label file
    rel = os.path.relpath(img_path, dataset_root)
    label_path = os.path.join(dataset_root, 'labels', os.path.relpath(img_path, val_rel))
    # The above may not match layout; try expected label path: replace images/ with labels/
    label_path = img_path
    label_path = label_path.replace(os.path.join('images', 'val'), os.path.join('labels', 'val'))
    label_txt = os.path.splitext(label_path)[0] + '.txt'

    # load ground truth boxes
    gt_boxes = []
    if os.path.exists(label_txt):
        with open(label_txt, 'r') as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(float(parts[0]))
                    # YOLO format x_center y_center w h (relative)
                    x_c = float(parts[1]); y_c = float(parts[2]); w_rel = float(parts[3]); h_rel = float(parts[4])
                    # read image size
                    img = cv2.imread(img_path)
                    h_img, w_img = img.shape[:2]
                    x1 = int((x_c - w_rel/2.0) * w_img)
                    y1 = int((y_c - h_rel/2.0) * h_img)
                    x2 = int((x_c + w_rel/2.0) * w_img)
                    y2 = int((y_c + h_rel/2.0) * h_img)
                    gt_boxes.append({'class': cls, 'box':[x1,y1,x2,y2]})
    else:
        # no labels -> skip
        continue

    # run prediction for this image
    results = model(img_path, conf=conf, iou=0.5)
    res = results[0]
    pred_boxes = []
    if res.boxes is not None:
        for box in res.boxes:
            conf_score = float(box.conf[0])
            cls = int(box.cls[0])
            coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
            pred_boxes.append({'class':cls, 'box':coords, 'conf':conf_score})

    # match preds to gts
    matched_gt = set()
    matched_pred = set()
    for pi, p in enumerate(pred_boxes):
        best_iou = 0.0
        best_gi = -1
        for gi, g in enumerate(gt_boxes):
            # only match same class
            if p['class'] != g['class']:
                continue
            # compute IoU
            x1 = max(p['box'][0], g['box'][0]); y1 = max(p['box'][1], g['box'][1])
            x2 = min(p['box'][2], g['box'][2]); y2 = min(p['box'][3], g['box'][3])
            if x2<=x1 or y2<=y1:
                iou = 0.0
            else:
                inter = (x2-x1)*(y2-y1)
                area_p = (p['box'][2]-p['box'][0])*(p['box'][3]-p['box'][1])
                area_g = (g['box'][2]-g['box'][0])*(g['box'][3]-g['box'][1])
                union = area_p + area_g - inter
                iou = inter/union if union>0 else 0.0
            if iou > best_iou:
                best_iou = iou; best_gi = gi
        if best_iou >= IoU_thr and best_gi not in matched_gt:
            matched_gt.add(best_gi); matched_pred.add(pi)
            summary['TP'] += 1
        else:
            summary['FP'] += 1
    # any unmatched gt are FN
    fn = 0
    for gi in range(len(gt_boxes)):
        if gi not in matched_gt:
            fn += 1
    summary['FN'] += fn

    if fn>0 or (len(pred_boxes)-len(matched_pred))>0:
        failure_cases.append({'image': img_path, 'GT': len(gt_boxes), 'Pred': len(pred_boxes), 'TP': len(matched_pred), 'FP': len(pred_boxes)-len(matched_pred), 'FN': fn})

# Print summary
print('\nPer-image match summary (IoU=0.5):')
print(json.dumps(summary, indent=2))
print('\nFailure case count:', len(failure_cases))
# Save failure cases to file
out = os.path.join('results', 'eval_failure_cases.json')
os.makedirs('results', exist_ok=True)
with open(out, 'w') as f:
    json.dump({'summary':summary, 'failures': failure_cases[:200]}, f, indent=2)
print('Wrote failure cases to', out)
