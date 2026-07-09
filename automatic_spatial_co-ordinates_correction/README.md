# Automatic Spatial Coordinates Correction Pipeline

This tool automates the process of correcting and calibrating camera spatial coordinates (`pixels_per_mm`) using a feedback loop based on known coffee bean grades.

## Pipeline Steps
1. **Initial Grading Pass (Step 1)**: Runs the YOLOv8 and SAM 2 (optional) software on every photo in the target folder and writes details to a CSV.
2. **Difference Calculation (Step 2)**: Compares the average measured length with the standard range for the target grade (parsed dynamically from `grading/Grade.txt`) and calculates differences.
3. **Spatial Calibration Feedback Loop (Steps 3-5)**: Iteratively adjusts the spatial template dimensions (`TEMPLATE_WIDTH_MM` and `TEMPLATE_HEIGHT_MM` configuration on the detector) in minute increments until the calculated average length falls into the expected range and classifies as the target grade.
4. **Validation Pass (Step 6)**: Runs a full verification pass using the calibrated settings for each photo.
5. **Folder Audit (Step 7)**: Verifies that the folder-wide average is correct, alerting if re-calibration is needed.

---

## 🏷️ Complexity Categorization (For Model Selection)

### 🟢 The Easy Part
- **Directory & File Operations**: Scanning the target folder, listing images, saving and loading adjustments to `adjustments.json`.
- **Parsing Configuration & Grade Bounds**: Parsing the `grading/Grade.txt` file using regex or split matching to extract minimum and maximum grade boundaries.
- **Reporting & Data Exporting**: Building rows and writing statistical calculations (such as average, standard deviation, and bounding differences) to a CSV (`results_correction.csv`).

### 🔴 The Hard Part (Requires Advanced Reasoning / Gemini 1.5 Pro or similar)
- **Deep Learning Model Interfacing**: Executing YOLOv8 detector inference, generating segmentation masks with SAM 2, and extracting key-point features (midline crease lengths or box contours) dynamically.
- **Feedback Loop Controller (Convergent Search)**: Constructing a robust PID-like or step-wise feedback controller that correctly modifies the physical template size configuration on the object detector, ensuring that changes converge to a valid grade without oscillating or falling into infinite loops.
- **Calibration Scaling Math**: Aligning the relationship between physical pixel sizes, adjusted templates, average lengths, and OpenCV threshold boundaries.

---

## 🚀 How to Run (Wait for Confirmation to Execute)

### Step 1: Place Your Images
Create a folder called `A grade` (either in the project root or in `data/raw/`) and put all your "A grade" images inside.

### Step 2: Run the Pipeline CLI
```bash
python automatic_spatial_co-ordinates_correction/pipeline.py --folder "A grade" --grade "A"
```

Options:
- `--folder <path>`: Folder of images (default: `A grade`).
- `--grade <grade>`: Target grade to calibrate against (default: `A`).
- `--use-sam2`: If specified, uses high-fidelity SAM 2 masks for length calculations instead of Otsu threshold contours.
- `--max-iter <num>`: Max adjustment iterations per image (default: `50`).
- `--step-size <num>`: Minute step adjustment factor (default: `0.005`).
