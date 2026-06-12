# ☕ Coffee Bean Detection & Analysis

Detect, count, grade, and analyze coffee beans using a custom YOLOv8 model. Includes a Flask web app for instant image upload, annotated preview, screen grading, quality classification, and ArcFace-based front/back bean mapping.

## ✨ Features

- **Pre-Analysis Guide** — Step-by-step photo instructions (white calibration template, 350g sample, front & back shots)
- **Dual Image Upload** — Upload front-facing and back-facing bean photos separately
- **Bean Detection & Counting** — Custom YOLOv8 model with contour fallback
 - **Defect Classification** — Black, broken, foreign, moldy, overfermented, Type A, and other defect types
- **Screen Grading** — Maps bean sizes to standard screen numbers (13–20) with aperture sizes, counts, and percentages
- **Quality Grade (AAA→C)** — Assigns a grade based on defect count (AAA = ≤3 defects, C = 60+)
- **Weight & Density** — Enter 350g sample weight → avg weight per bean calculated automatically
 - **Size Dimensions** — Avg length, width, L/W ratio, size class distribution (auto-calibrated from the white template, coin optional)
- **Color Analysis** — Color distribution and pixel-level color picker
- **ArcFace Mapping** — Front↔back bean matching infrastructure using ResNet-18 embeddings (swappable with trained ArcFace model)
- **Web Interface** — Modern responsive Flask app with drag-and-drop upload

## 🚀 Quick Start

```powershell
# 1. Clone the repo
git clone https://github.com/Bhagat2678/coffee-bean.git
cd coffee-bean

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the web app
python website/app.py
# → Open http://localhost:5000
```

> **Note:** The model file `models/best.pt` is not stored in git (large binary). Get it from the team or train your own — see [Training](#training) below.
> Without it, the app falls back to `yolov8n.pt` (COCO-pretrained, auto-downloaded on first run).

## 📁 Project Structure

```
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
├── src/
│   ├── detector.py           # YOLOv8 detection engine
│   ├── analyzer.py           # Color & size analysis
│   ├── grading.py            # Screen grading, density, AAA→C classification
│   ├── arcface.py            # Front-back bean mapping (ArcFace infrastructure)
│   └── cvat_converter.py     # CVAT XML → YOLO format converter
├── models/
│   └── best.pt               # Custom YOLOv8n model (not in git — see note above)
├── website/
│   ├── app.py                # Flask backend
│   ├── templates/index.html  # Frontend UI
│   ├── static/css/style.css  # Styles
│   ├── static/js/app.js      # Frontend logic
│   └── uploads/              # Runtime upload folder (auto-created)
├── database/
│   └── schema.sql            # PostgreSQL schema (optional)
├── scripts/
│   ├── train.py              # Train the model
│   ├── evaluate.py           # Evaluate metrics
│   ├── infer.py              # Run inference
│   └── add_dataset.py        # Add new datasets
├── data/
│   ├── raw/                  # Input images (not in git)
│   └── output/               # Annotated outputs (not in git)
└── runs/                     # YOLO training runs (not in git)
```

## 📸 How to Use the Web App

1. **Weigh** — Use exactly 350g of coffee beans (or enter your weight)
2. **Spread** — Spread beans evenly on the white calibration template, no overlapping
3. **Photograph** — Take a well-lit top-down photo of the front side
4. **Flip & Photograph** — Flip all beans and take a second photo of the back side
5. **Upload** — Drop both images into the Front and Back upload zones
6. **Analyze** — Click Analyze to get:
   - Bean count, non-bean count
   - Quality grade (AAA → C)
   - Screen grading table (Screen 13–20)
   - Avg weight per bean (density)
   - Size dimensions (requires the white calibration template or a coin for mm calibration)
   - Color distribution
   - Defect breakdown

> **Important:** For accurate mm-size calibration, ensure the **white template/background** is visible in the image.
> The system automatically detects the template (113.52mm × 180.41mm) and uses it for calibration.
> **Fallback:** If the template isn't detected, place a **5-rupee coin** (23mm diameter) in the frame for calibration.

## 📊 Model Performance

Custom YOLOv8n trained on 5,505 images (3 defect classes).

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
# Retrain with current dataset
python scripts/train.py --epochs 150 --batch 16

# Add a new dataset first, then retrain
python scripts/add_dataset.py <path_to_dataset>
python scripts/train.py --epochs 150 --name v2
```

## 🔗 ArcFace Bean Mapping

The `src/arcface.py` module maps individual beans from the front image to their counterparts in the back image:

1. **Current state** — Uses ResNet-18 (ImageNet features) for coarse matching
2. **Collect data** — Every front+back analysis auto-saves paired crops to `website/static/pairs/`
3. **Train model** — Train a dedicated ArcFace model on the collected pairs
4. **Swap backbone** — Place trained model at `models/arcface_beans.pt` and it's auto-loaded

## 🌐 API

```
POST /analyze
  Body: multipart/form-data
    front_image   (required) — front-facing beans image
    back_image    (optional) — back-facing beans image
    sample_weight (optional, default 350) — sample weight in grams

  Response JSON:
    bean_count, non_bean_count, object_count
    grade: { grade, label, defect_count, defect_percentage, defect_breakdown }
    density: { sample_weight_g, bean_count, avg_weight_per_bean_g }
    screen_distribution: [ { screen, aperture_mm, count, percentage }, ... ]
    size_stats: { avg_length_mm, avg_width_mm, avg_lw_ratio, ... }
    color_distribution, detections, annotated_image_url
    back_annotated_image_url (if back image uploaded)
    arcface_pairs (if both images uploaded)
```

## 📋 Requirements

- Python 3.10+
- PyTorch (with CUDA for GPU training — see [pytorch.org](https://pytorch.org))
- Ultralytics YOLOv8
- OpenCV, NumPy, SciPy
- Flask, Werkzeug
