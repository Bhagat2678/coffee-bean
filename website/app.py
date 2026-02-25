from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import uuid
import time
import sys

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.detector import ObjectDetector

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['ANNOTATED_FOLDER'] = os.path.join(app.static_folder, 'annotated')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANNOTATED_FOLDER'], exist_ok=True)

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


def _get_request_float(name, default):
    try:
        val = request.form.get(name, request.args.get(name, None))
        return float(val) if val is not None else default
    except Exception:
        return default


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
    uid = uuid.uuid4().hex
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{uid}.{ext}')
    file.save(upload_path)

    start = time.time()
    try:
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
    except Exception as e:
        return jsonify({'error': 'Server error during inference', 'details': str(e)}), 500
    end = time.time()

    detections = result.get('detections', [])
    avg_conf = sum(d.get('confidence', 0) for d in detections) / len(detections) if detections else 0.0

    response = {
        'bean_count': result.get('bean_count', result.get('total_count', 0)),
        'non_bean_count': result.get('non_bean_count', 0),
        'object_count': result.get('total_count', 0),
        'processing_time': round((end - start) * 1000, 1),
        'annotated_image_url': url_for('static', filename=f'annotated/{annotated_name}'),
        'detection_source': result.get('detection_source', 'model'),
        'color_distribution': result.get('color_distribution', {}),
        'avg_bean_size_mm': result.get('avg_bean_size_mm'),
        'detections': detections
    }

    return jsonify(response)


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
