# ☕ Coffee Bean Detection & Analysis

Detect, count, and analyze coffee beans in images using YOLOv8 object detection. Includes a Flask web app for instant image upload and visual analysis.

## ✨ Features

- **Bean Detection & Counting** — Accurately detects and counts individual coffee beans using a custom-trained YOLOv8 model
- **Non-Bean Object Filtering** — Identifies and separates non-bean objects (coins, debris) using color/shape heuristics
- **Defect Classification** — Classifies beans by defect type: black, broken, and foreign matter
- **Color & Size Analysis** — Extracts color distribution and estimated bean sizes
- **Web Interface** — Modern, responsive Flask web app with drag-and-drop image upload, annotated image preview, and detailed results
- **CLI Mode** — Command-line interface for batch processing and scripting
- **Contour Fallback** — Automatic contour-based detection when model confidence is low

## 🚀 Quick Start

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run Web App
python website/app.py                   # open http://localhost:5000

# Run CLI
python main.py                          # interactive mode
python main.py my_image.jpg             # direct mode (image in data/raw/)
```

## 📁 Project Structure

```
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── detector.py         # YOLOv8 detection engine (ObjectDetector class)
│   ├── analyzer.py         # Color & size analysis (BeanAnalyzer class)
│   └── cvat_converter.py   # CVAT XML → YOLO format converter
├── models/
│   └── best.pt             # Custom trained YOLOv8n model
├── website/
│   ├── app.py              # Flask backend (routes: /, POST /analyze)
│   ├── templates/
│   │   └── index.html      # Frontend UI
│   ├── static/
│   │   ├── css/            # Stylesheets
│   │   ├── js/             # Frontend logic
│   │   └── img/            # Static assets
│   └── uploads/            # Uploaded images (created at runtime)
├── datasets/
│   └── consolidated/       # Training data (5,505 images)
│       └── dataset.yaml    # Dataset configuration
├── scripts/
│   ├── train.py            # Train the model
│   ├── evaluate.py         # Evaluate model metrics
│   ├── infer.py            # Run inference + analysis
│   └── add_dataset.py      # Add new training datasets
├── notebooks/
│   └── 01_experiment.ipynb # Experiment notebook
├── data/
│   ├── raw/                # Input images
│   └── output/             # Annotated output images
└── runs/                   # YOLO training runs & logs
```

## 🌐 Web App

The web interface provides an easy way to analyze coffee bean images:

1. **Upload** — Drag and drop or select an image (JPG, PNG)
2. **Analyze** — The backend runs YOLOv8 inference with post-processing
3. **Results** — View annotated image, bean count, non-bean count, color distribution, and processing time

API endpoint:
```
POST /analyze
  Body: multipart/form-data with field "image"
  Response: JSON with bean_count, non_bean_count, annotated_image_url, color_distribution, detections, etc.
```

## 📊 Model Performance

Custom YOLOv8n trained on 5,505 images (3 defect classes) with an RTX 4060.

| Metric     | Score |
|------------|-------|
| mAP50      | 0.967 |
| mAP50-95   | 0.756 |
| Precision  | 0.961 |
| Recall     | 0.942 |

| Class   | Precision | Recall | mAP50 |
|---------|-----------|--------|-------|
| black   | 0.974     | 0.977  | 0.991 |
| broken  | 0.978     | 0.928  | 0.940 |
| foreign | 0.930     | 0.922  | 0.970 |

## 🏋️ Training

```powershell
# Retrain with current data
python scripts/train.py --epochs 150 --batch 16

# Add a new dataset then retrain
python scripts/add_dataset.py <path_to_dataset>
python scripts/train.py --epochs 150 --name v2
```

## 📋 Requirements

- Python 3.10+
- PyTorch (with CUDA for GPU training)
- Ultralytics YOLOv8
- OpenCV, NumPy
- Flask, Werkzeug