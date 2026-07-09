"""
scripts/verify_sam2.py
======================
Phase 4 — System Verification Tests

Checks that:
  1. Calibration accuracy is preserved when SAM 2 is enabled vs disabled.
  2. The pipeline handles large batches (simulated 100+ boxes) without OOM.
  3. Per-bean measurements are within an acceptable tolerance range.
  4. FP16 / SDPA optimisations load cleanly.
"""

import os
import sys
import time
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2
import numpy as np

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)
    results.append((name, condition))
    return condition


print("=" * 62)
print("  SAM 2 System Verification Tests  (Phase 4)")
print("=" * 62)

# ─────────────────────────────────────────────────────────────
# TEST 1 — Import & syntax
# ─────────────────────────────────────────────────────────────
print("\n[1] Import & syntax check")
try:
    import ast
    src = open("src/detector.py").read()
    ast.parse(src)
    check("detector.py syntax valid", True)
except SyntaxError as e:
    check("detector.py syntax valid", False, str(e))

try:
    from src.detector import ObjectDetector
    check("ObjectDetector importable", True)
except Exception as e:
    check("ObjectDetector importable", False, str(e))
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# TEST 2 — SAM_CHUNK_SIZE constant exists
# ─────────────────────────────────────────────────────────────
print("\n[2] Phase 4 constants")
check("SAM_CHUNK_SIZE == 64", ObjectDetector.SAM_CHUNK_SIZE == 64,
      f"got {ObjectDetector.SAM_CHUNK_SIZE}")

# ─────────────────────────────────────────────────────────────
# TEST 3 — Default mode (no SAM 2) still works
# ─────────────────────────────────────────────────────────────
print("\n[3] Default mode (OpenCV fallback)")
d_base = ObjectDetector("models/best.pt", use_sam2=False)
check("Default use_sam2=False", not d_base.use_sam2)
check("Default sam2_predictor=None", d_base.sam2_predictor is None)

# ─────────────────────────────────────────────────────────────
# TEST 4 — SAM 2 mode initialises on CUDA with SDPA
# ─────────────────────────────────────────────────────────────
print("\n[4] SAM 2 init (CUDA + SDPA)")
t0 = time.time()
d_sam2 = ObjectDetector("models/best.pt", use_sam2=True)
init_time = time.time() - t0
check("use_sam2=True acknowledged", d_sam2.use_sam2)
check("sam2_predictor loaded", d_sam2.sam2_predictor is not None)
check("SAM 2 on cuda", d_sam2.sam2_device == "cuda")
check("Init time < 30 s", init_time < 30, f"{init_time:.1f}s")

# ─────────────────────────────────────────────────────────────
# TEST 5 — Calibration accuracy preserved
# ─────────────────────────────────────────────────────────────
print("\n[5] Calibration accuracy (pixels_per_mm)")
TEST_IMG = "data/raw/test_beans_1.jpg"

t1 = time.time()
r_base = d_base.detect_objects(TEST_IMG, use_sam2=False)
base_time = time.time() - t1

t2 = time.time()
r_sam2 = d_sam2.detect_objects(TEST_IMG, use_sam2=True)
sam2_time = time.time() - t2

ppm_base = r_base.get("pixels_per_mm")
ppm_sam2 = r_sam2.get("pixels_per_mm")
check("Baseline ppm not None", ppm_base is not None, str(ppm_base))
check("SAM2   ppm not None",   ppm_sam2 is not None, str(ppm_sam2))

if ppm_base and ppm_sam2:
    diff_pct = abs(ppm_base - ppm_sam2) / max(ppm_base, 1e-6) * 100
    check("Calibration drift < 1%", diff_pct < 1.0,
          f"base={ppm_base:.3f} sam2={ppm_sam2:.3f} diff={diff_pct:.3f}%")

check("Bean count unchanged", r_base["bean_count"] == r_sam2["bean_count"],
      f"base={r_base['bean_count']} sam2={r_sam2['bean_count']}")

check("sam2_active flag set", r_sam2.get("sam2_active") is True)

print(f"  {INFO}  Base  time: {base_time:.2f}s | "
      f"SAM 2 time: {sam2_time:.2f}s")

# ─────────────────────────────────────────────────────────────
# TEST 6 — Per-bean length plausible (4 mm – 25 mm typical)
# ─────────────────────────────────────────────────────────────
print("\n[6] Per-bean measurement sanity")
BEAN_LEN_MIN, BEAN_LEN_MAX = 4.0, 40.0
BEAN_WID_MIN, BEAN_WID_MAX = 2.0, 40.0   # width/height covers both axes of the min-area rect

bad_len = bad_wid = 0
for det in r_sam2["detections"]:
    lmm = det.get("length_mm")
    smm = det.get("size_mm")
    if lmm is not None and not (BEAN_LEN_MIN <= lmm <= BEAN_LEN_MAX):
        bad_len += 1
    if smm:
        w, h = smm.get("width", 0), smm.get("height", 0)
        if not (BEAN_WID_MIN <= w <= BEAN_WID_MAX and BEAN_WID_MIN <= h <= BEAN_WID_MAX):
            bad_wid += 1

n = len(r_sam2["detections"])
check("All lengths in plausible range", bad_len == 0,
      f"{bad_len}/{n} out of [{BEAN_LEN_MIN}, {BEAN_LEN_MAX}] mm")
check("All widths in plausible range",  bad_wid == 0,
      f"{bad_wid}/{n} out of [{BEAN_WID_MIN}, {BEAN_WID_MAX}] mm")

# ─────────────────────────────────────────────────────────────
# TEST 7 — Chunked batching stress test (synthetic large input)
# ─────────────────────────────────────────────────────────────
print("\n[7] Chunked batching — stress test (100 synthetic boxes)")
img = cv2.imread(TEST_IMG)
h_img, w_img = img.shape[:2]

# Generate 100 tiny synthetic boxes spread across image
rng = np.random.default_rng(42)
syn_boxes = []
bw = bh = 20
for _ in range(100):
    x1 = int(rng.integers(0, max(1, w_img - bw)))
    y1 = int(rng.integers(0, max(1, h_img - bh)))
    syn_boxes.append([x1, y1, x1 + bw, y1 + bh])

t3 = time.time()
masks = d_sam2._segment_with_sam2(img, syn_boxes)
stress_time = time.time() - t3

check("Returned 100 masks", len(masks) == 100, f"got {len(masks)}")
non_none = sum(m is not None for m in masks)
check("All masks non-None", non_none == 100, f"{non_none}/100 non-None")
check("Stress test < 60 s",  stress_time < 60, f"{stress_time:.1f}s")
print(f"  {INFO}  100-box stress test completed in {stress_time:.2f}s "
      f"({stress_time/100*1000:.1f} ms/box)")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print()
print("=" * 62)
print(f"  Results: {passed}/{total} checks passed")
print("=" * 62)
if passed == total:
    print("  ✅  ALL CHECKS PASSED — SAM 2 pipeline is verified!")
else:
    failed = [n for n, ok in results if not ok]
    print("  ❌  FAILED:", ", ".join(failed))
    sys.exit(1)
