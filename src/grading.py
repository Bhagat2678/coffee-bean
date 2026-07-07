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
    {"screen": 19.5, "aperture_mm": 7.75},
    {"screen": 19, "aperture_mm": 7.50},
    {"screen": 18.5, "aperture_mm": 7.25},
    {"screen": 18, "aperture_mm": 7.00},
    {"screen": 17, "aperture_mm": 6.75},
    {"screen": 16, "aperture_mm": 6.50},
    {"screen": 15, "aperture_mm": 6.00},
    {"screen": 14, "aperture_mm": 5.50},
    {"screen": 13, "aperture_mm": 5.25},
    {"screen": 12, "aperture_mm": 5.00},
    {"screen": 11, "aperture_mm": 4.50},
    {"screen": 10, "aperture_mm": 4.00},
    {"screen": 9, "aperture_mm": 3.50},
    {"screen": 8, "aperture_mm": 3.00},
]

# Defect classes that the YOLO model can report
DEFECT_CLASSES = {"black", "broken", "immature", "infested", "sour", 
                 "overfermented", "moldy", "defective", "damaged", 
                 "fermented", "underripe", "premature", "fungal", "diseased",
                 "infected"}


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
    # Build a zero-count dict for every screen + "Below 8"
    screen_counts = {s["screen"]: 0 for s in SCREEN_TABLE}
    screen_counts["Below 8"] = 0

    bean_detections = [d for d in detections
                       if d.get("object_type") == "coffee_bean" or d.get("class") == "coffee_bean"]

    total = len(bean_detections)
    if total == 0:
        return _format_screen_result(screen_counts, total)

    for det in bean_detections:
        width_mm = _get_width_mm(det, pixels_per_mm)
        if width_mm is None:
            # Can't classify without size — put in "Below 8" as fallback
            screen_counts["Below 8"] += 1
            continue

        placed = False
        for row in SCREEN_TABLE:
            if width_mm >= row["aperture_mm"]:
                screen_counts[row["screen"]] += 1
                placed = True
                break
        if not placed:
            screen_counts["Below 8"] += 1

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
        # Determine grade for this screen
        grade_info = africa_india_grade_for_screen_64(row["screen"])
        result.append({
            "screen": row["screen"],
            "aperture_mm": row["aperture_mm"],
            "screen_64": row["screen"],
            "africa_india_grade": grade_info["grade"],
            "count": cnt,
            "percentage": pct,
        })
    # Below 8
    cnt = screen_counts["Below 8"]
    pct = round((cnt / total) * 100, 1) if total > 0 else 0.0
    grade_info_below_8 = africa_india_grade_for_screen_64(7.0)
    result.append({
        "screen": "Below 8",
        "aperture_mm": "< 3.00",
        "screen_64": "< 8",
        "africa_india_grade": grade_info_below_8["grade"],
        "count": cnt,
        "percentage": pct,
    })
    return result


def mm_to_screen_64(length_mm):
    """Convert millimetres to the coffee screen scale measured in 1/64 inch."""
    if length_mm is None:
        return None
    return round((float(length_mm) / 25.4) * 64, 2)


def africa_india_grade_for_screen_64(screen_64):
    """Map a screen size in 1/64 inch scale to the Indian grading system."""
    if screen_64 is None:
        return {"grade": "Triage", "label": "Indian Triage", "min_screen": None}

    screen_val = float(screen_64)
    # Using boundaries according to Grade.txt (Indian Grade Standard):
    # PB (Peaberry) : Screen 10-13
    # AA            : Screen 18+ (>= 17.5 mid-point boundary)
    # A             : Screen 17 (>= 16.5)
    # B             : Screen 16 (>= 15.5)
    # C             : Screen 15 (>= 14.5)
    # BB            : Screen 14 (>= 13.5)
    # Triage        : Screen < 14 (excluding PB 10-13)
    if screen_val >= 17.5:
        return {"grade": "AA", "label": "Indian AA", "min_screen": 18.0}
    elif screen_val >= 16.5:
        return {"grade": "A", "label": "Indian A", "min_screen": 17.0}
    elif screen_val >= 15.5:
        return {"grade": "B", "label": "Indian B", "min_screen": 16.0}
    elif screen_val >= 14.5:
        return {"grade": "C", "label": "Indian C", "min_screen": 15.0}
    elif screen_val >= 13.5:
        return {"grade": "BB", "label": "Indian BB", "min_screen": 14.0}
    elif 9.5 <= screen_val < 13.5:
        return {"grade": "PB", "label": "Indian PB", "min_screen": 10.0}
    else:
        return {"grade": "Triage", "label": "Indian Triage", "min_screen": None}


def africa_india_grade_for_length_mm(length_mm):
    """Map a bean/screen aperture in mm to the Indian grading system."""
    if length_mm is None:
        return {"grade": "Triage", "label": "Indian Triage", "min_screen": None}
    
    length_val = float(length_mm)
    # Using length boundaries directly from Grade.txt (Indian Grade Standard):
    # PB            : 3.97–5.16
    # AA            : ≥7.5
    # A             : 6.75–7.49
    # B             : 6.35–6.74
    # C             : 5.95–6.34
    # BB            : 5.56–5.94
    # Triage        : <5.56 (excluding PB)
    if length_val >= 7.5:
        return {"grade": "AA", "label": "Indian AA", "min_screen": 18.0}
    elif length_val >= 6.75:
        return {"grade": "A", "label": "Indian A", "min_screen": 17.0}
    elif length_val >= 6.35:
        return {"grade": "B", "label": "Indian B", "min_screen": 16.0}
    elif length_val >= 5.95:
        return {"grade": "C", "label": "Indian C", "min_screen": 15.0}
    elif length_val >= 5.56:
        return {"grade": "BB", "label": "Indian BB", "min_screen": 14.0}
    elif 3.97 <= length_val <= 5.16:
        return {"grade": "PB", "label": "Indian PB", "min_screen": 10.0}
    else:
        return {"grade": "Triage", "label": "Indian Triage", "min_screen": None}


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
    Assign a quality grade (AA → Triage) based on average bean length (mm)
    mapped to the Indian grading system.

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

    # Count defects for diagnostic details (e.g. breakdown, UI stats)
    defect_breakdown = {}
    defect_count = 0
    MIN_DEFECT_CONFIDENCE = 0.4
    for det in detections:
        # Only count defects for actual coffee beans (not non-beans)
        if det.get("object_type") != "coffee_bean" and det.get("class") != "coffee_bean":
            continue
        conf = float(det.get("confidence") or 0.0)
        if conf < MIN_DEFECT_CONFIDENCE:
            continue
        defect_type = (det.get("defect_type") or "").lower()
        if defect_type in DEFECT_CLASSES:
            defect_count += 1
            defect_breakdown[defect_type] = defect_breakdown.get(defect_type, 0) + 1

    defect_pct = round((defect_count / bean_count) * 100, 2) if bean_count > 0 else 0.0

    # Determine grade based on bean length (crease midline or max of size)
    bean_lengths = []
    for det in detections:
        if det.get("object_type") != "coffee_bean" and det.get("class") != "coffee_bean":
            continue
        
        # Prefer midline crease length if present
        length_mm = det.get("length_mm")
        if length_mm is not None:
            bean_lengths.append(float(length_mm))
        else:
            # Fallback to the larger dimension of size_mm
            sm = det.get("size_mm")
            if sm and sm.get("width") and sm.get("height"):
                l = max(float(sm["width"]), float(sm["height"]))
                bean_lengths.append(l)

    grade = "Triage"
    label = "Indian Triage"
    avg_length = None
    avg_screen_64 = None

    if bean_lengths:
        avg_length = sum(bean_lengths) / len(bean_lengths)
        avg_screen_64 = mm_to_screen_64(avg_length)
        grade_row = africa_india_grade_for_length_mm(avg_length)
        grade = grade_row["grade"]
        label = grade_row["label"]

    return {
        "grade": grade,
        "label": label,
        "grade_basis": "indian_screen_length",
        "screen_64": avg_screen_64,
        "avg_length_mm": round(avg_length, 2) if avg_length is not None else None,
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
        length_mm = det.get("length_mm")
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

        # Prefer detector-provided crease/midline length when available.
        # Fall back to the larger dimension of the size estimate if needed.
        length = float(length_mm) if length_mm else max(w, h)
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
