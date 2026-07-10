from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import uuid
import time
import sys
import cv2
import json
import numpy as np

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.detector import ObjectDetector
from src.grading import (
    compute_screen_distribution,
    compute_density,
    classify_grade,
    compute_size_stats,
)

# Calibration constants for 4-corner perspective rectification
TEMPLATE_WIDTH_MM = 180.13
TEMPLATE_HEIGHT_MM = 113.4
PIXELS_PER_MM = 12.0
OUTPUT_WIDTH_PX = round(TEMPLATE_WIDTH_MM * PIXELS_PER_MM)   # 2162
OUTPUT_HEIGHT_PX = round(TEMPLATE_HEIGHT_MM * PIXELS_PER_MM) # 1361

def rectify_image(image_path, src_corners, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image {image_path} for perspective warping.")
        return False
    
    src_pts = np.array(src_corners, dtype=np.float32)
    
    # Calculate measured edge lengths (TL -> TR and TL -> BL)
    edge_top = np.linalg.norm(src_pts[1] - src_pts[0])   # TL -> TR
    edge_left = np.linalg.norm(src_pts[3] - src_pts[0])  # TL -> BL
    
    print(f"[DEBUG] Perspective Rectification:")
    print(f"  edge_top (TL->TR): {edge_top:.2f} px")
    print(f"  edge_left (TL->BL): {edge_left:.2f} px")
    ratio = edge_top / edge_left if edge_left > 0 else 0
    print(f"  Ratio (top/left): {ratio:.3f} (expected landscape ~1.59, portrait ~0.63)")
    
    if edge_top >= edge_left:
        # Landscape: top edge is the long physical edge (180.13 mm)
        out_w, out_h = OUTPUT_WIDTH_PX, OUTPUT_HEIGHT_PX
        print(f"  Orientation determined: LANDSCAPE -> target {out_w}x{out_h}")
    else:
        # Portrait: top edge is the short physical edge (113.4 mm)
        out_w, out_h = OUTPUT_HEIGHT_PX, OUTPUT_WIDTH_PX
        print(f"  Orientation determined: PORTRAIT -> target {out_w}x{out_h}")
        
    # Target coordinates: TL, TR, BR, BL order (matches frontend click sequence)
    dst_pts = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified = cv2.warpPerspective(image, matrix, (out_w, out_h))
    cv2.imwrite(output_path, rectified)
    return True

# ArcFace mapper — optional, gracefully degrade
try:
    from src.arcface import ArcFaceMapper
    ARCFACE_AVAILABLE = True
except ImportError:
    ARCFACE_AVAILABLE = False

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['ANNOTATED_FOLDER'] = os.path.join(app.static_folder, 'annotated')
app.config['PAIRS_FOLDER'] = os.path.join(app.static_folder, 'pairs')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANNOTATED_FOLDER'], exist_ok=True)
os.makedirs(app.config['PAIRS_FOLDER'], exist_ok=True)

# Model configuration — single best model
MODEL_PATH = os.path.join(ROOT, 'models', 'best.pt')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8n.pt"
    print("[WARN] models/best.pt not found, using yolov8n.pt")
else:
    print(f"[INFO] Using model: models/best.pt")

# Detection defaults
CONFIDENCE = 0.25
MIN_CONF_OUTPUT = 0.25
MIN_BOX_AREA_RATIO = 0.0005
MAX_BOX_AREA_RATIO = 0.7
MIN_ASPECT = 0.25
MAX_ASPECT = 4.0
IOU = 0.3
BOX_SHRINK_RATIO = 0.02

# Global detector instance (initialized with SAM 2 support)
DETECTOR = ObjectDetector(MODEL_PATH, use_sam2=True)

# Load adjustments config
ADJUSTMENTS_PATH = os.path.join(ROOT, 'automatic_spatial_co-ordinates_correction', 'adjustments.json')
ADJUSTMENTS = {}
if os.path.exists(ADJUSTMENTS_PATH):
    try:
        with open(ADJUSTMENTS_PATH, 'r') as f:
            ADJUSTMENTS = json.load(f)
        print(f"[INFO] Loaded adjustments for {len(ADJUSTMENTS)} images from adjustments.json")
    except Exception as e:
        print(f"[WARN] Failed to load adjustments.json: {e}")

# ArcFace mapper instance (lazy)
_arcface_mapper = None


def _get_arcface():
    global _arcface_mapper
    if _arcface_mapper is None and ARCFACE_AVAILABLE:
        arcface_model = os.path.join(ROOT, 'models', 'arcface_beans.pt')
        model_path = arcface_model if os.path.exists(arcface_model) else None
        _arcface_mapper = ArcFaceMapper(model_path=model_path)
    return _arcface_mapper


def _get_request_float(name, default):
    try:
        val = request.form.get(name, request.args.get(name, None))
        return float(val) if val is not None else default
    except Exception:
        return default


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _run_detection(upload_path, uid, ext, manual_crop=False, rectified=False, pixels_per_mm=None, filename=None):
    """Run YOLO detection on a single image. Returns (result_dict, annotated_url)."""
    # Reset detector template size to baseline class defaults
    DETECTOR.TEMPLATE_WIDTH_MM = ObjectDetector.TEMPLATE_WIDTH_MM
    DETECTOR.TEMPLATE_HEIGHT_MM = ObjectDetector.TEMPLATE_HEIGHT_MM

    # If this specific original image has an adjustment scale, apply it
    if filename and filename in ADJUSTMENTS:
        scale = ADJUSTMENTS[filename]
        DETECTOR.TEMPLATE_WIDTH_MM = ObjectDetector.TEMPLATE_WIDTH_MM * scale
        DETECTOR.TEMPLATE_HEIGHT_MM = ObjectDetector.TEMPLATE_HEIGHT_MM * scale
        print(f"[INFO] Applied adjustments scale {scale} for {filename}")

    annotated_name = f'{uid}_annotated.{ext}'
    annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], annotated_name)

    req_conf = _get_request_float('confidence_threshold', CONFIDENCE)
    req_min_out = _get_request_float('min_confidence_output', req_conf)
    req_iou = _get_request_float('iou', IOU)

    req_use_sam2 = request.form.get('use_sam2', 'true') == 'true'

    result = DETECTOR.detect_objects(
        upload_path,
        confidence_threshold=req_conf,
        save_output=annotated_path,
        iou=req_iou,
        min_confidence_output=req_min_out,
        min_box_area_ratio=MIN_BOX_AREA_RATIO,
        max_box_area_ratio=MAX_BOX_AREA_RATIO,
        min_aspect=MIN_ASPECT,
        max_aspect=MAX_ASPECT,
        box_shrink_ratio=BOX_SHRINK_RATIO,
        manual_crop=manual_crop,
        use_sam2=req_use_sam2,
        debug=False,
        rectified=rectified,
        pixels_per_mm=pixels_per_mm
    )

    annotated_url = url_for('static', filename=f'annotated/{annotated_name}')
    return result, annotated_url


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    # ── Accept front image (required) and back image (optional) ──
    front_file = request.files.get('front_image') or request.files.get('image')
    back_file = request.files.get('back_image')

    if not front_file or front_file.filename == '':
        return jsonify({'error': 'No front image provided'}), 400
    if not allowed_file(front_file.filename):
        return jsonify({'error': 'Invalid file type for front image'}), 400

    # Save front image
    ext_front = secure_filename(front_file.filename).rsplit('.', 1)[1].lower()
    uid_front = uuid.uuid4().hex
    upload_front = os.path.join(app.config['UPLOAD_FOLDER'], f'{uid_front}.{ext_front}')
    front_file.save(upload_front)

    # Save back image (if provided)
    upload_back = None
    uid_back = None
    ext_back = None
    if back_file and back_file.filename != '' and allowed_file(back_file.filename):
        ext_back = secure_filename(back_file.filename).rsplit('.', 1)[1].lower()
        uid_back = uuid.uuid4().hex
        upload_back = os.path.join(app.config['UPLOAD_FOLDER'], f'{uid_back}.{ext_back}')
        back_file.save(upload_back)

    # Sample weight (default 350g)
    sample_weight = _get_request_float('sample_weight', 350.0)

    # Parse corners if provided by the frontend
    front_corners = None
    back_corners = None
    
    front_corners_str = request.form.get('front_corners')
    if front_corners_str:
        try:
            front_corners = json.loads(front_corners_str)
        except Exception as e:
            print(f"[WARN] Failed to parse front_corners: {e}")

    back_corners_str = request.form.get('back_corners')
    if back_corners_str:
        try:
            back_corners = json.loads(back_corners_str)
        except Exception as e:
            print(f"[WARN] Failed to parse back_corners: {e}")

    start = time.time()
    
    front_rectified = False
    front_detect_path = upload_front
    if front_corners and len(front_corners) == 4:
        rectified_front_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{uid_front}_rectified.{ext_front}')
        if rectify_image(upload_front, front_corners, rectified_front_path):
            front_detect_path = rectified_front_path
            front_rectified = True

    back_rectified = False
    back_detect_path = upload_back
    if upload_back and back_corners and len(back_corners) == 4:
        rectified_back_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{uid_back}_rectified.{ext_back}')
        if rectify_image(upload_back, back_corners, rectified_back_path):
            back_detect_path = rectified_back_path
            back_rectified = True

    try:
        # ── Run detection on FRONT image ──
        if front_rectified:
            # Compensate for inner compartment mismatch (stretching)
            # Default scale correction factor of 0.75 makes pixels_per_mm 12.0 / 0.75 = 16.0
            rect_ppm = PIXELS_PER_MM / 0.75
            front_result, front_annotated_url = _run_detection(
                front_detect_path, uid_front, ext_front, rectified=True, pixels_per_mm=rect_ppm, filename=front_file.filename
            )
        else:
            print("[WARN] Running old auto-calibration code path because no corners were provided.")
            front_result, front_annotated_url = _run_detection(
                front_detect_path, uid_front, ext_front, manual_crop=False, filename=front_file.filename
            )

        # ── Run detection on BACK image (if provided) ──
        back_result = None
        back_annotated_url = None
        if upload_back:
            if back_rectified:
                rect_ppm = PIXELS_PER_MM / 0.75
                back_result, back_annotated_url = _run_detection(
                    back_detect_path, uid_back, ext_back, rectified=True, pixels_per_mm=rect_ppm, filename=back_file.filename
                )
            else:
                back_result, back_annotated_url = _run_detection(
                    back_detect_path, uid_back, ext_back, manual_crop=False, filename=back_file.filename
                )

    except Exception as e:
        return jsonify({'error': 'Server error during inference', 'details': str(e)}), 500
    end = time.time()

    # ── Primary counts come from front image ──
    detections = front_result.get('detections', [])
    bean_count = front_result.get('bean_count', 0)
    non_bean_count = front_result.get('non_bean_count', 0)
    total_count = front_result.get('total_count', 0)
    pixels_per_mm = front_result.get('pixels_per_mm')

    # ── Grading computations ──
    density_info = compute_density(sample_weight, bean_count)
    screen_dist = compute_screen_distribution(detections, pixels_per_mm)
    grade_info = classify_grade(detections, bean_count)
    size_stats = compute_size_stats(detections, pixels_per_mm)


    # ── ArcFace front-back matching (if both images provided) ──
    arcface_pairs = []
    if upload_back and back_result:
        mapper = _get_arcface()
        if mapper:
            try:
                front_img = cv2.imread(front_detect_path)
                back_img = cv2.imread(back_detect_path)
                front_dets = front_result.get('detections', [])
                back_dets = back_result.get('detections', [])

                front_embs = mapper.extract_embeddings(front_img, front_dets)
                back_embs = mapper.extract_embeddings(back_img, back_dets)
                arcface_pairs = mapper.match_front_back(front_embs, back_embs, threshold=0.3)

                # Save pairs for training
                if arcface_pairs:
                    pair_dir = os.path.join(app.config['PAIRS_FOLDER'], uid_front)
                    mapper.save_paired_dataset(front_img, back_img, arcface_pairs,
                                               front_dets, back_dets, pair_dir)
            except Exception as e:
                print(f"[WARN] ArcFace matching failed: {e}")
                arcface_pairs = []

    # ── Build response ──
    response = {
        'bean_count': bean_count,
        'non_bean_count': non_bean_count,
        'object_count': total_count,
        'processing_time': round((end - start) * 1000, 1),
        'annotated_image_url': front_annotated_url,
        'detection_source': front_result.get('detection_source', 'model'),
        'color_distribution': front_result.get('color_distribution', {}),
        'avg_bean_size_mm': front_result.get('avg_bean_size_mm'),
        'avg_bean_length_mm': front_result.get('avg_bean_length_mm'),
        'detections': detections,
        # New fields
        'sample_weight_g': sample_weight,
        'density': density_info,
        'screen_distribution': screen_dist,
        'grade': grade_info,
        'size_stats': size_stats,
        'pixels_per_mm': pixels_per_mm,
        'pixels_per_mm_x': front_result.get('pixels_per_mm_x'),
        'pixels_per_mm_y': front_result.get('pixels_per_mm_y'),
    }

    # Back image fields
    if back_result:
        response['back_annotated_image_url'] = back_annotated_url
        response['back_bean_count'] = back_result.get('bean_count', 0)

    # ArcFace pairs (strip numpy arrays for JSON)
    if arcface_pairs:
        clean_pairs = []
        for p in arcface_pairs:
            clean_pairs.append({
                'front_index': p['front_index'],
                'back_index': p['back_index'],
                'similarity': p['similarity'],
            })
        response['arcface_pairs'] = clean_pairs
        response['arcface_pair_count'] = len(clean_pairs)

    return jsonify(response)


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
