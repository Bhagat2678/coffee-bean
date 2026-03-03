#!/usr/bin/env python3
"""
Run detection with a saved YOLO model and analyze detections with `BeanAnalyzer` if available.

Usage:
  python scripts/run_infer_analyze.py --image path/to/image.jpg [--model models/name.pt] [--conf 0.25]

Outputs:
 - Prints a short summary to stdout (total detections, avg size, simple color stats).
 - Saves an annotated image to `runs/infer/<timestamp>/annotated.jpg`.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


def find_default_model():
    models_dir = ROOT / 'models'
    if not models_dir.exists():
        return None
    pts = sorted(models_dir.glob('*.pt'))
    return str(pts[0]) if pts else None


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    return img


def annotate_and_save(img, boxes, scores, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    a = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(a, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{i}:{scores[i]:.2f}'
        cv2.putText(a, label, (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    out_path = Path(out_dir) / 'annotated.jpg'
    cv2.imwrite(str(out_path), a)
    return str(out_path)


def basic_color_stats(img, box):
    x1, y1, x2, y2 = map(int, box)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return {'mean_bgr': [0,0,0]}
    mean = cv2.mean(crop)[:3]
    return {'mean_bgr': [float(mean[0]), float(mean[1]), float(mean[2])]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True, help='Path to input image')
    p.add_argument('--model', default=None, help='Path to model .pt (defaults to first in models/)')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--device', default=None, help="Device to run on, e.g. 'cpu' or '0' for GPU. If omitted auto-detects")
    p.add_argument('--output-json', default=None, help='Path to save JSON summary')
    p.add_argument('--save', action='store_true', help='Save annotated image (default: True)')
    p.add_argument('--no-save', dest='save', action='store_false')
    p.set_defaults(save=True)
    p.add_argument('--summary-only', action='store_true', help='Print concise summary (count, color distribution, size distribution)')
    args = p.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f'Image not found: {img_path}')

    model_path = args.model or find_default_model()
    if model_path is None:
        raise SystemExit('No model specified and no models/*.pt found in `models/`.')

    if YOLO is None:
        raise SystemExit('Ultralytics YOLO package not available in the environment.')

    print('Using model:', model_path)
    model = YOLO(str(model_path))

    # choose device: CLI override -> auto-detect
    device = args.device
    if device is None:
        try:
            import torch
            device = 0 if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'

    # run prediction with augmentation for better accuracy
    print('Device:', device)
    print('Running detection (augment=True, imgsz=1280). This may take a moment...')
    results = model.predict(source=str(img_path), imgsz=1280, conf=args.conf, iou=0.45, augment=True, device=device, verbose=False)
    if len(results) == 0:
        print('No results returned by model.')
        return
    res = results[0]

    # extract boxes and scores robustly
    boxes = []
    scores = []
    if hasattr(res, 'boxes') and res.boxes is not None:
        try:
            # ultralytics result boxes may have .xyxy and .conf
            xyxy = getattr(res.boxes, 'xyxy', None)
            confs = getattr(res.boxes, 'conf', None)
            if xyxy is None:
                xyxy = getattr(res.boxes, 'xyxy', None)
            if xyxy is not None:
                arr = xyxy.cpu().numpy()
                boxes = arr.tolist()
            if confs is not None:
                scores = confs.cpu().numpy().tolist()
        except Exception:
            pass

    # fallback: try res.boxes.xyxy if present as tensor
    if not boxes:
        try:
            arr = res.boxes.xyxy.cpu().numpy()
            boxes = arr.tolist()
        except Exception:
            boxes = []

    if not scores and hasattr(res, 'probs'):
        try:
            scores = res.probs.cpu().numpy().tolist()
        except Exception:
            scores = []

    # ensure scores list matches boxes
    if len(scores) < len(boxes):
        scores = scores + [0.0] * (len(boxes) - len(scores))

    img = load_image(img_path)

    # Try to use BeanAnalyzer from src.analyzer if available
    analyzer_summary = None
    try:
        from src.analyzer import BeanAnalyzer
        analyzer = BeanAnalyzer()
        # try a few possible analyzer method names
        if hasattr(analyzer, 'analyze'):
            analyzer_summary = analyzer.analyze(img, boxes)
        elif hasattr(analyzer, 'analyze_from_numpy_image'):
            analyzer_summary = analyzer.analyze_from_numpy_image(img, boxes)
        else:
            analyzer_summary = None
    except Exception:
        analyzer_summary = None

    # If analyzer not available or failed, compute a basic summary
    basic_summary = {}
    basic_summary['total_detections'] = len(boxes)
    areas = []
    per_box_color = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        areas.append(max(0, (x2 - x1) * (y2 - y1)))
        per_box_color.append(basic_color_stats(img, box))
    basic_summary['avg_area'] = float(np.mean(areas)) if areas else 0.0
    basic_summary['per_box_color'] = per_box_color

    summary = {'model': str(model_path), 'image': str(img_path), 'basic': basic_summary}
    if analyzer_summary is not None:
        summary['analyzer'] = analyzer_summary

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = ROOT / 'runs' / 'infer' / f'infer_{ts}'
    annotated_path = None
    if args.save:
        annotated_path = annotate_and_save(img, boxes, scores, out_dir)
        summary['annotated_image'] = annotated_path

    # print to stdout
    print(json.dumps(summary, indent=2))

    # optionally save JSON summary to disk
    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f'Saved JSON summary: {outp}')

    # If user requested concise summary, print minimal info and exit
    if args.summary_only:
        # prefer analyzer summary when available
        counts = 0
        color_dist = {}
        size_dist = {}
        if analyzer_summary is not None:
            counts = analyzer_summary.get('total_count', analyzer_summary.get('count', 0))
            color_dist = analyzer_summary.get('color_distribution', analyzer_summary.get('colors', {}))
            size_dist = analyzer_summary.get('size_distribution', analyzer_summary.get('sizes', {}))
        else:
            # fallback: derive from basic_summary
            counts = basic_summary.get('total_detections', 0)
            # compute color buckets from per_box_color
            def map_color(bgr):
                b, g, r = [int(x) for x in bgr]
                px = np.uint8([[[b, g, r]]])
                hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
                h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
                if 5 <= h <= 30:
                    return 'Orange-Brown'
                if 31 <= h <= 70:
                    return 'Yellow-Green'
                if 71 <= h <= 150:
                    return 'Green'
                if (h >= 151 and h <= 179) or (h >= 0 and h <= 4):
                    return 'Red-ish'
                return 'Other'

            color_counts = {}
            for c in basic_summary.get('per_box_color', []):
                bgr = c.get('mean_bgr', [0, 0, 0])
                label = map_color(bgr)
                color_counts[label] = color_counts.get(label, 0) + 1
            color_dist = color_counts

            # size distribution by relative area
            img_h, img_w = img.shape[0], img.shape[1]
            total_pixels = img_h * img_w
            small = mid = large = 0
            areas_list = areas if 'areas' in locals() else []
            for a in areas_list:
                rel = a / total_pixels if total_pixels else 0
                if rel < 0.005:
                    small += 1
                elif rel < 0.02:
                    mid += 1
                else:
                    large += 1
            size_dist = {'Small': small, 'Medium': mid, 'Large': large}

        # print concise lines
        print(f'Total: {counts}')
        print('Color distribution: ' + json.dumps(color_dist))
        print('Size distribution: ' + json.dumps(size_dist))


if __name__ == '__main__':
    main()
