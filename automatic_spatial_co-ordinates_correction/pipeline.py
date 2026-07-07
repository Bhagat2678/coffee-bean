import os
import sys
import json
import csv
import argparse
from pathlib import Path

# Add project root to sys.path to allow imports from src/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.detector import ObjectDetector
from src.grading import classify_grade

# Base template dimensions defined in ObjectDetector class
BASE_TEMPLATE_WIDTH = 113.52
BASE_TEMPLATE_HEIGHT = 180.41

def get_grade_range(grade_char, grade_txt_path):
    """
    Parses Grade.txt to extract the min and max length (mm) for the given grade character.
    E.g., A -> (6.75, 7.49)
    """
    defaults = {
        "PB": (3.97, 5.16),
        "AA": (7.5, 99.0),
        "A": (6.75, 7.49),
        "B": (6.35, 6.74),
        "C": (5.95, 6.34),
        "BB": (5.56, 5.94),
        "Triage": (0.0, 5.55)
    }
    
    if not os.path.exists(grade_txt_path):
        print(f"[INFO] {grade_txt_path} not found. Using default grade ranges.")
        return defaults.get(grade_char.upper(), (6.75, 7.49))
        
    min_len, max_len = None, None
    try:
        with open(grade_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    grade_name = parts[1]
                    length_str = parts[3]
                    if grade_name.upper() == grade_char.upper() or grade_name.split()[0].upper() == grade_char.upper():
                        # Parse length string: e.g., "6.75–7.49", "≥7.5", or "<5.56"
                        length_str = length_str.replace("–", "-").replace("—", "-")
                        if "-" in length_str:
                            low_s, high_s = length_str.split("-")
                            min_len = float(low_s.strip())
                            max_len = float(high_s.strip())
                        elif "≥" in length_str or ">=" in length_str:
                            clean_s = length_str.replace("≥", "").replace(">=", "")
                            min_len = float(clean_s.strip())
                            max_len = 99.0
                        elif "<" in length_str:
                            clean_s = length_str.replace("<", "")
                            min_len = 0.0
                            max_len = float(clean_s.strip()) - 0.01
                        break
    except Exception as e:
        print(f"[WARN] Error parsing {grade_txt_path}: {e}. Using default.")
        
    if min_len is None or max_len is None:
        return defaults.get(grade_char.upper(), (6.75, 7.49))
    return min_len, max_len

def calculate_average_length(detections):
    """Calculate the average length of coffee beans from detections."""
    bean_lengths = []
    for det in detections:
        if det.get("object_type") == "coffee_bean" or det.get("class") == "coffee_bean":
            length_mm = det.get("length_mm")
            if length_mm is not None:
                bean_lengths.append(float(length_mm))
            else:
                sm = det.get("size_mm")
                if sm and sm.get("width") and sm.get("height"):
                    bean_lengths.append(max(float(sm["width"]), float(sm["height"])))
    if not bean_lengths:
        return 0.0
    return sum(bean_lengths) / len(bean_lengths)

def compute_difference(avg_len, min_len, max_len):
    """
    Calculate the difference from the target range.
    If avg_len is below min_len, difference is negative (avg_len - min_len).
    If avg_len is above max_len, difference is positive (avg_len - max_len).
    Otherwise, difference is 0.0 (inside range).
    """
    if avg_len < min_len:
        return round(avg_len - min_len, 3)
    elif avg_len > max_len:
        return round(avg_len - max_len, 3)
    else:
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Automatic Spatial Coordinates Correction Pipeline")
    parser.add_argument("--folder", type=str, default="A grade", help="Path to the folder of photos")
    parser.add_argument("--grade", type=str, default="A", help="Target grade (e.g. A, AA, PB, B)")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLOv8 model file")
    parser.add_argument("--use-sam2", action="store_true", help="Enable SAM 2 segmentation")
    parser.add_argument("--max-iter", type=int, default=50, help="Maximum adjustment iterations per image")
    parser.add_argument("--step-size", type=float, default=0.005, help="Step size for minute template adjustments")
    parser.add_argument("--initial-only", action="store_true", help="Only run initial grading pass and export CSV, skipping adjustments loop")
    parser.add_argument("--skip-validation", action="store_true", help="Run corrections (Steps 3-5) but skip the folder-wide validation pass (Steps 6-7)")
    args = parser.parse_args()

    # Determine input directory path
    folder_path = Path(args.folder)
    if not folder_path.exists():
        # Check inside data/raw
        fallback_path = ROOT_DIR / "data" / "raw" / args.folder
        if fallback_path.exists():
            folder_path = fallback_path
        else:
            print(f"❌ Error: Grade folder '{args.folder}' does not exist.")
            print(f"Please create it at '{folder_path.resolve()}' or '{fallback_path.resolve()}' containing the grade images.")
            sys.exit(1)

    print(f"📁 Target Folder: {folder_path.resolve()}")
    print(f"🏷️ Target Grade: {args.grade}")
    
    # Get grade range
    grade_txt_path = ROOT_DIR / "grading" / "Grade.txt"
    min_len, max_len = get_grade_range(args.grade, grade_txt_path)
    print(f"📐 Target Length Range for Grade '{args.grade}': {min_len}mm - {max_len}mm")

    # Load model
    model_path = args.model
    if not os.path.exists(model_path):
        model_path = str(ROOT_DIR / "yolov8n.pt")
        print(f"[WARN] Model file '{args.model}' not found. Falling back to default YOLOv8 nano: {model_path}")
    
    # Initialize ObjectDetector
    detector = ObjectDetector(model_path, use_sam2=args.use_sam2)

    # Get list of images
    image_extensions = ('.jpg', '.jpeg', '.png')
    images = [f for f in folder_path.iterdir() if f.suffix.lower() in image_extensions]
    if not images:
        print(f"❌ Error: No images found in '{folder_path}'")
        sys.exit(1)
        
    print(f"📸 Found {len(images)} images to process.")

    output_dir = Path(__file__).resolve().parent
    results_csv_path = output_dir / "results_correction.csv"
    adjustments_json_path = output_dir / "adjustments.json"

    # Load existing adjustments if any (to build on top or review)
    adjustments = {}
    if adjustments_json_path.exists():
        try:
            with open(adjustments_json_path, "r") as f:
                adjustments = json.load(f)
            print(f"ℹ️ Loaded existing scale factors for {len(adjustments)} images.")
        except Exception as e:
            print(f"[WARN] Error loading existing adjustments: {e}")

    # CSV headers
    csv_headers = [
        "image_name",
        "total_beans",
        "original_pixels_per_mm",
        "original_avg_length_mm",
        "original_grade",
        "adjusted_pixels_per_mm",
        "adjusted_avg_length_mm",
        "adjusted_grade",
        "scale_factor",
        "diff_from_min",
        "diff_from_max",
        "difference",
        "iterations",
        "status"
    ]

    csv_rows = []
    
    print("\n🚀 Starting Step 1-5: Running software and making adjustments per image...")
    for idx, img_path in enumerate(images, 1):
        img_name = img_path.name
        print(f"\n[{idx}/{len(images)}] Processing image: {img_name}")

        # Reset template parameters to baseline
        detector.TEMPLATE_WIDTH_MM = BASE_TEMPLATE_WIDTH
        detector.TEMPLATE_HEIGHT_MM = BASE_TEMPLATE_HEIGHT

        # --- Initial Run (Step 1) ---
        res = detector.detect_objects(str(img_path), confidence_threshold=0.25, use_sam2=args.use_sam2)
        detections = res.get('detections', [])
        total_beans = sum(1 for d in detections if d.get('object_type') == 'coffee_bean')
        orig_ppm = res.get('pixels_per_mm') or 0.0
        orig_avg_len = calculate_average_length(detections)
        orig_grade_info = classify_grade(detections)
        orig_grade = orig_grade_info['grade']

        print(f"  • Initial Run: Beans = {total_beans}, Avg Length = {orig_avg_len:.2f}mm, Grade = {orig_grade}, Pixels/mm = {orig_ppm:.2f}")

        # Check if already in range
        if args.initial_only:
            print(f"  ℹ️ Running in --initial-only mode. Skipping adjustments.")
            scale_factor = 1.0
            adj_ppm = orig_ppm
            adj_avg_len = orig_avg_len
            adj_grade = orig_grade
            iterations = 0
            status = "Initial Pass Only"
        elif orig_grade == args.grade and min_len <= orig_avg_len <= max_len:
            print(f"  ✅ Image is already in '{args.grade}' range. No adjustment needed.")
            scale_factor = 1.0
            adj_ppm = orig_ppm
            adj_avg_len = orig_avg_len
            adj_grade = orig_grade
            iterations = 0
            status = "Already OK"
        else:
            if orig_avg_len == 0.0 or orig_ppm == 0.0:
                print(f"  ❌ Calibration failed (No template/plate detected). Skipping adjustments.")
                scale_factor = 1.0
                adj_ppm = orig_ppm
                adj_avg_len = orig_avg_len
                adj_grade = orig_grade
                iterations = 0
                status = "Failed (No template)"
                adjustments[img_name] = scale_factor
            else:
                print(f"  ⚠️ Grade or length mismatch. Starting iterative adjustments (Step 3)...")
                scale_factor = adjustments.get(img_name, 1.0)
                iterations = 0
                success = False
                adj_ppm = orig_ppm
                adj_avg_len = orig_avg_len
                adj_grade = orig_grade

                # Minute adjustments loop (Step 3 & Step 4)
                for iter_count in range(1, args.max_iter + 1):
                    # Apply the adjusted scale factor to template size configuration
                    detector.TEMPLATE_WIDTH_MM = BASE_TEMPLATE_WIDTH * scale_factor
                    detector.TEMPLATE_HEIGHT_MM = BASE_TEMPLATE_HEIGHT * scale_factor

                    res_adj = detector.detect_objects(str(img_path), confidence_threshold=0.25, use_sam2=args.use_sam2)
                    dets_adj = res_adj.get('detections', [])
                    adj_ppm = res_adj.get('pixels_per_mm') or 0.0
                    adj_avg_len = calculate_average_length(dets_adj)
                    adj_grade_info = classify_grade(dets_adj)
                    adj_grade = adj_grade_info['grade']

                    print(f"    - Iteration {iter_count}: Scale = {scale_factor:.4f}, Avg Length = {adj_avg_len:.2f}mm, Grade = {adj_grade}")

                    if adj_avg_len == 0.0 or adj_ppm == 0.0:
                        print(f"    ❌ Length or PPM became 0.0. Breaking adjustment loop.")
                        break

                    # Check if it satisfies requirements
                    if adj_grade == args.grade and min_len <= adj_avg_len <= max_len:
                        print(f"    ✅ Success on iteration {iter_count}! Adjusted Scale = {scale_factor:.4f}")
                        success = True
                        iterations = iter_count
                        status = "Corrected"
                        break

                    # Approach target length range:
                    if adj_avg_len < min_len:
                        scale_factor += args.step_size
                    elif adj_avg_len > max_len:
                        scale_factor -= args.step_size
                    else:
                        print(f"    ⚠️ Warning: Length {adj_avg_len:.2f}mm in bounds, but grade '{adj_grade}' mismatch.")
                        break

                if not success:
                    print(f"  ❌ Failed to correct coordinates within {args.max_iter} iterations.")
                    status = "Failed"
                    iterations = args.max_iter

                # Save the computed scale factor
                adjustments[img_name] = scale_factor

        # Differences (Step 2)
        diff_min = round(adj_avg_len - min_len, 3)
        diff_max = round(max_len - adj_avg_len, 3)
        diff_overall = compute_difference(adj_avg_len, min_len, max_len)

        csv_rows.append({
            "image_name": img_name,
            "total_beans": total_beans,
            "original_pixels_per_mm": round(orig_ppm, 3) if orig_ppm else 0.0,
            "original_avg_length_mm": round(orig_avg_len, 3),
            "original_grade": orig_grade,
            "adjusted_pixels_per_mm": round(adj_ppm, 3) if adj_ppm else 0.0,
            "adjusted_avg_length_mm": round(adj_avg_len, 3),
            "adjusted_grade": adj_grade,
            "scale_factor": round(scale_factor, 4),
            "diff_from_min": diff_min,
            "diff_from_max": diff_max,
            "difference": diff_overall,
            "iterations": iterations,
            "status": status
        })

    # Save CSV (Step 1 & Step 2)
    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n📝 CSV results saved to: {results_csv_path}")

    # Save scale adjustments config
    with open(adjustments_json_path, "w") as f:
        json.dump(adjustments, f, indent=4)
    print(f"⚙️ Scale adjustments config saved to: {adjustments_json_path}")

    if args.initial_only:
        print("\n🎉 Initial pass complete. Exiting as --initial-only was specified.")
        sys.exit(0)

    if args.skip_validation:
        print("\n✅ Steps 3-5 complete. Calibration adjustments saved.")
        print("📂 CSV results       : ", results_csv_path)
        print("⚙️  Scale factors JSON : ", adjustments_json_path)
        print("\n⏸️  PAUSED — Please switch to your model for Steps 6-7 (folder-wide validation).")
        print("   Then re-run WITHOUT --skip-validation to run the final validation pass.")
        sys.exit(0)

    # --- Step 6: Run the entire software on the whole folder again ---
    print("\n🔄 Starting Step 6: Validating the entire folder using saved adjustments...")
    all_valid = True
    validated_lengths = []
    
    for idx, img_path in enumerate(images, 1):
        img_name = img_path.name
        scale = adjustments.get(img_name, 1.0)
        
        # Apply the calibrated scale factor for validation run
        detector.TEMPLATE_WIDTH_MM = BASE_TEMPLATE_WIDTH * scale
        detector.TEMPLATE_HEIGHT_MM = BASE_TEMPLATE_HEIGHT * scale

        res = detector.detect_objects(str(img_path), confidence_threshold=0.25, use_sam2=args.use_sam2)
        detections = res.get('detections', [])
        avg_len = calculate_average_length(detections)
        grade_info = classify_grade(detections)
        grade = grade_info['grade']
        validated_lengths.append(avg_len)

        in_range = min_len <= avg_len <= max_len
        print(f"  • {img_name}: Length = {avg_len:.2f}mm, Grade = {grade} -> {'✅ OK' if (grade == args.grade and in_range) else '❌ OUT OF RANGE'}")
        
        if grade != args.grade or not in_range:
            all_valid = False

    # Calculate overall stats
    overall_avg = sum(validated_lengths) / len(validated_lengths) if validated_lengths else 0.0
    print("\n📊 Folder Validation Summary:")
    print(f"  - Overall Average Bean Length: {overall_avg:.2f}mm")
    print(f"  - Target Grade '{args.grade}' Range: {min_len}mm - {max_len}mm")
    print(f"  - All images within range: {'✅ Yes' if all_valid else '❌ No'}")

    # --- Step 7: Check if the average is correct ---
    if not all_valid or not (min_len <= overall_avg <= max_len):
        print("\n❌ Step 7: Folder-wide verification FAILED. Some images or the overall average is not correct.")
        print("Please review logs/adjustments, check image calibration plates, or re-run the pipeline.")
        sys.exit(1)
    else:
        print("\n🎉 Success! Automatic grade correction pipeline is fully calibrated and verified.")

if __name__ == "__main__":
    main()
