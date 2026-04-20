"""
Module: grading.py
Description: Coffee bean quality grading — screen distribution, density,
             abnormality classification, and size statistics.
"""

# ---------------------------------------------------------------------------
# Screen Grading Table  (Indian standard)
# Screen number → minimum aperture in mm   (beans that stay ON the sieve)
# ---------------------------------------------------------------------------
SCREEN_TABLE = [
    {"screen": 20, "aperture_mm": 8.00},
    {"screen": 19, "aperture_mm": 7.50},
    {"screen": 18, "aperture_mm": 7.10},
    {"screen": 17, "aperture_mm": 6.70},
    {"screen": 16, "aperture_mm": 6.30},
    {"screen": 15, "aperture_mm": 5.95},
    {"screen": 14, "aperture_mm": 5.60},
    {"screen": 13, "aperture_mm": 5.00},
]

# Grade definitions: grade → max defect count (inclusive)
GRADE_THRESHOLDS = [
    {"grade": "AAA", "label": "Specialty",     "max_defects": 3},
    {"grade": "AA",  "label": "Premium",       "max_defects": 8},
    {"grade": "A",   "label": "Good",          "max_defects": 20},
    {"grade": "B",   "label": "Fair / Average", "max_defects": 40},
    {"grade": "BA",  "label": "Below Average",  "max_defects": 60},
    {"grade": "C",   "label": "Commercial",     "max_defects": None},  # everything above 60
]

# Defect classes that the YOLO model can report
DEFECT_CLASSES = {"black", "broken", "foreign", "immature", "infested", "sour", "overfermented", "moldy"}


# ── Screen Grading ─────────────────────────────────────────────────────────

def compute_screen_distribution(detections, pixels_per_mm=None):
    """
    Map each detected bean's width (mm) to a screen number.

    Args:
        detections (list[dict]): Each dict must have 'box' [x1,y1,x2,y2]
                                 and optionally 'size_mm' {'width': …, 'height': …}.
        pixels_per_mm (float|None): Pixel-to-mm calibration factor.

    Returns:
        list[dict]: One entry per screen, each with
                    {'screen', 'aperture_mm', 'count', 'percentage'}.
                    Sorted from largest (Screen 20) to smallest.
    """
    # Build a zero-count dict for every screen + "Below 13"
    screen_counts = {s["screen"]: 0 for s in SCREEN_TABLE}
    screen_counts["Below 13"] = 0

    bean_detections = [d for d in detections
                       if d.get("object_type") == "coffee_bean" or d.get("class") == "coffee_bean"]

    total = len(bean_detections)
    if total == 0:
        return _format_screen_result(screen_counts, total)

    for det in bean_detections:
        width_mm = _get_width_mm(det, pixels_per_mm)
        if width_mm is None:
            # Can't classify without size — put in "Below 13" as fallback
            screen_counts["Below 13"] += 1
            continue

        placed = False
        for row in SCREEN_TABLE:
            if width_mm >= row["aperture_mm"]:
                screen_counts[row["screen"]] += 1
                placed = True
                break
        if not placed:
            screen_counts["Below 13"] += 1

    return _format_screen_result(screen_counts, total)


def _get_width_mm(det, pixels_per_mm):
    """Extract bean width in mm from detection dict."""
    # Prefer pre-computed size_mm
    sm = det.get("size_mm")
    if sm and sm.get("width"):
        return float(sm["width"])

    # Compute from box + calibration
    if pixels_per_mm and pixels_per_mm > 0:
        box = det.get("box")
        if box:
            bw = max(1, box[2] - box[0])
            bh = max(1, box[3] - box[1])
            # Use the larger dimension as the bean length, smaller as width
            w_mm = round(min(bw, bh) / pixels_per_mm, 2)
            return w_mm
    return None


def _format_screen_result(screen_counts, total):
    result = []
    for row in SCREEN_TABLE:
        cnt = screen_counts[row["screen"]]
        pct = round((cnt / total) * 100, 1) if total > 0 else 0.0
        result.append({
            "screen": row["screen"],
            "aperture_mm": row["aperture_mm"],
            "count": cnt,
            "percentage": pct,
        })
    # Below 13
    cnt = screen_counts["Below 13"]
    pct = round((cnt / total) * 100, 1) if total > 0 else 0.0
    result.append({
        "screen": "Below 13",
        "aperture_mm": "< 5.00",
        "count": cnt,
        "percentage": pct,
    })
    return result


# ── Density Calculator ─────────────────────────────────────────────────────

def compute_density(sample_weight_g, bean_count):
    """
    Average density (weight per bean).

    Args:
        sample_weight_g (float): Total sample weight in grams.
        bean_count (int): Number of beans detected.

    Returns:
        dict: {'sample_weight_g', 'bean_count', 'avg_weight_per_bean_g'}
    """
    if bean_count <= 0:
        return {
            "sample_weight_g": sample_weight_g,
            "bean_count": 0,
            "avg_weight_per_bean_g": None,
        }

    avg = round(sample_weight_g / bean_count, 4)
    return {
        "sample_weight_g": sample_weight_g,
        "bean_count": bean_count,
        "avg_weight_per_bean_g": avg,
    }


# ── Abnormality Classification ─────────────────────────────────────────────

def classify_grade(detections, bean_count=None):
    """
    Assign a quality grade (AAA → C) based on defect count.

    Defects are detected beans whose model label (defect_type) indicates
    a problem: black, broken, foreign, etc.

    Args:
        detections (list[dict]): Enriched detections from the detector.
        bean_count (int|None): Override total bean count.

    Returns:
        dict: {'grade', 'label', 'defect_count', 'defect_percentage',
               'defect_breakdown', 'total_beans'}
    """
    if bean_count is None:
        bean_count = sum(
            1 for d in detections
            if d.get("object_type") == "coffee_bean" or d.get("class") == "coffee_bean"
        )

    defect_breakdown = {}
    defect_count = 0
    for det in detections:
        defect_type = (det.get("defect_type") or "").lower()
        if defect_type in DEFECT_CLASSES:
            defect_count += 1
            defect_breakdown[defect_type] = defect_breakdown.get(defect_type, 0) + 1

    defect_pct = round((defect_count / bean_count) * 100, 2) if bean_count > 0 else 0.0

    grade = "C"
    label = "Commercial"
    for entry in GRADE_THRESHOLDS:
        if entry["max_defects"] is None:
            grade = entry["grade"]
            label = entry["label"]
            break
        if defect_count <= entry["max_defects"]:
            grade = entry["grade"]
            label = entry["label"]
            break

    return {
        "grade": grade,
        "label": label,
        "defect_count": defect_count,
        "defect_percentage": defect_pct,
        "defect_breakdown": defect_breakdown,
        "total_beans": bean_count,
    }


# ── Size Statistics ────────────────────────────────────────────────────────

def compute_size_stats(detections, pixels_per_mm=None):
    """
    Compute fine-tuned size statistics for detected beans.

    Returns:
        dict: {'avg_length_mm', 'avg_width_mm', 'avg_lw_ratio',
               'min_length_mm', 'max_length_mm',
               'min_width_mm', 'max_width_mm',
               'size_class_distribution'}
    """
    bean_dets = [d for d in detections
                 if d.get("object_type") == "coffee_bean" or d.get("class") == "coffee_bean"]

    lengths = []
    widths = []

    for det in bean_dets:
        sm = det.get("size_mm")
        if sm and sm.get("width") and sm.get("height"):
            w = float(sm["width"])
            h = float(sm["height"])
        elif pixels_per_mm and pixels_per_mm > 0:
            box = det.get("box")
            if not box:
                continue
            bw = max(1, box[2] - box[0])
            bh = max(1, box[3] - box[1])
            w = round(bw / pixels_per_mm, 2)
            h = round(bh / pixels_per_mm, 2)
        else:
            continue

        # Length = larger dimension, width = smaller
        length = max(w, h)
        width = min(w, h)
        lengths.append(length)
        widths.append(width)

    if not lengths:
        return {
            "avg_length_mm": None,
            "avg_width_mm": None,
            "avg_lw_ratio": None,
            "min_length_mm": None,
            "max_length_mm": None,
            "min_width_mm": None,
            "max_width_mm": None,
            "bean_count_measured": 0,
            "size_class_distribution": {},
        }

    avg_l = round(sum(lengths) / len(lengths), 2)
    avg_w = round(sum(widths) / len(widths), 2)
    avg_ratio = round(avg_l / avg_w, 2) if avg_w > 0 else 0

    # Size class distribution based on length
    size_classes = {"Very Small (< 5mm)": 0, "Small (5-6mm)": 0, "Medium (6-7mm)": 0,
                    "Large (7-8mm)": 0, "Very Large (> 8mm)": 0}
    for l in lengths:
        if l < 5:
            size_classes["Very Small (< 5mm)"] += 1
        elif l < 6:
            size_classes["Small (5-6mm)"] += 1
        elif l < 7:
            size_classes["Medium (6-7mm)"] += 1
        elif l < 8:
            size_classes["Large (7-8mm)"] += 1
        else:
            size_classes["Very Large (> 8mm)"] += 1

    return {
        "avg_length_mm": avg_l,
        "avg_width_mm": avg_w,
        "avg_lw_ratio": avg_ratio,
        "min_length_mm": round(min(lengths), 2),
        "max_length_mm": round(max(lengths), 2),
        "min_width_mm": round(min(widths), 2),
        "max_width_mm": round(max(widths), 2),
        "bean_count_measured": len(lengths),
        "size_class_distribution": {k: v for k, v in size_classes.items() if v > 0},
    }
