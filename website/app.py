from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import uuid
import time
import sys
import cv2

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
BOX_SHRINK_RATIO = 0.15

# Global detector instance
DETECTOR = ObjectDetector(MODEL_PATH)

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


def _run_detection(upload_path, uid, ext):
    """Run YOLO detection on a single image. Returns (result_dict, annotated_url)."""
    annotated_name = f'{uid}_annotated.{ext}'
    annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], annotated_name)

    req_conf = _get_request_float('confidence_threshold', CONFIDENCE)
    req_min_out = _get_request_float('min_confidence_output', req_conf)
    req_iou = _get_request_float('iou', IOU)

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
        debug=False
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

    start = time.time()
    try:
        # ── Run detection on FRONT image ──
        front_result, front_annotated_url = _run_detection(upload_front, uid_front, ext_front)

        # ── Run detection on BACK image (if provided) ──
        back_result = None
        back_annotated_url = None
        if upload_back:
            back_result, back_annotated_url = _run_detection(upload_back, uid_back, ext_back)

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
                front_img = cv2.imread(upload_front)
                back_img = cv2.imread(upload_back)
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
