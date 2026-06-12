"""
Module: shape_analyzer.py
Description: Midline shape analysis, curvature measurement, and symmetry scoring
             for individual coffee bean masks.

Accepts a binary mask (single bean, white-on-black) and pixel-to-mm calibration
factors derived from the known tray dimensions (180 mm × 114 mm).  Returns
physical measurements and a shape profile used downstream by BeanClassifier.

Shape Profiles
--------------
STRAIGHT        — curvature_angle < 5°
SLIGHTLY_CURVED — 5° ≤ curvature_angle < 15°
CURVED          — 15° ≤ curvature_angle < 30°
IRREGULAR       — curvature_angle ≥ 30° OR symmetry_score < 0.50
"""

import math
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List


# ── Shape profile thresholds ──────────────────────────────────────────────
_PROFILE_STRAIGHT_MAX_DEG = 5.0
_PROFILE_SLIGHTLY_CURVED_MAX_DEG = 15.0
_PROFILE_CURVED_MAX_DEG = 30.0
_SYMMETRY_IRREGULAR_THRESHOLD = 0.50

# Minimum contour area (in pixels) to attempt analysis
_MIN_CONTOUR_AREA_PX = 25


class ShapeAnalyzer:
    """
    Analyse the shape of a single coffee bean from its binary mask.

    The mask must be a single-channel ``np.uint8`` image where the bean
    region is white (255) and the background is black (0).  Calibration
    factors ``px_per_mm_x`` and ``px_per_mm_y`` convert pixel measurements
    to real-world millimetres.

    Typical usage::

        analyzer = ShapeAnalyzer()
        result = analyzer.analyze(mask, px_per_mm_x=12.5, px_per_mm_y=12.3)
        print(result["profile"])  # e.g. "SLIGHTLY_CURVED"
    """

    # ── Public API ─────────────────────────────────────────────────────

    def analyze(
        self,
        mask: np.ndarray,
        px_per_mm_x: float,
        px_per_mm_y: float,
    ) -> Dict:
        """
        Run full shape analysis on a binary bean mask.

        Parameters
        ----------
        mask : np.ndarray
            Single-channel uint8 image (255 = bean, 0 = background).
        px_per_mm_x : float
            Horizontal pixels per millimetre (from tray calibration).
        px_per_mm_y : float
            Vertical pixels per millimetre (from tray calibration).

        Returns
        -------
        dict
            ``midline_length_mm``   – length along the bean's major axis (mm).
            ``max_width_mm``        – maximum width perpendicular to major axis (mm).
            ``curvature_angle_deg`` – angular deviation of midline from straight (°).
            ``symmetry_score``      – 0.0–1.0 mirror symmetry about the major axis.
            ``profile``             – one of STRAIGHT / SLIGHTLY_CURVED / CURVED / IRREGULAR.
        """
        if mask is None or mask.size == 0:
            return self._empty_result()

        # Ensure binary uint8
        mask_bin = self._ensure_binary(mask)

        # Largest contour
        contour = self._largest_contour(mask_bin)
        if contour is None or cv2.contourArea(contour) < _MIN_CONTOUR_AREA_PX:
            return self._empty_result()

        # Fit ellipse → principal axes (needs ≥ 5 points)
        if len(contour) < 5:
            return self._empty_result()

        ellipse = cv2.fitEllipse(contour)
        center, (minor_len, major_len), angle = ellipse
        # OpenCV fitEllipse: (width, height) where width ≤ height is not guaranteed;
        # the larger value is the major axis.
        if minor_len > major_len:
            minor_len, major_len = major_len, minor_len
            angle = (angle + 90.0) % 180.0

        # Align contour so major axis is horizontal, centred at origin
        aligned_contour, aligned_mask = self._align_contour(
            contour, mask_bin, center, angle
        )

        # Midline sampling along aligned major axis
        midline_points, upper_dists, lower_dists, span_px = self._sample_midline(
            aligned_contour, aligned_mask
        )

        # Curvature angle from midline deviation
        curvature_deg = self._compute_curvature_angle(midline_points, span_px)

        # Symmetry from upper/lower boundary distances
        symmetry = self._compute_symmetry(upper_dists, lower_dists)

        # Convert pixel measurements to mm (use average of x/y scale for diagonal)
        px_per_mm_avg = (abs(px_per_mm_x) + abs(px_per_mm_y)) / 2.0
        if px_per_mm_avg <= 0:
            px_per_mm_avg = 1.0  # safety fallback — result will be in pixels

        midline_length_mm = round(major_len / px_per_mm_avg, 2)
        max_width_mm = round(minor_len / px_per_mm_avg, 2)
        curvature_deg = round(curvature_deg, 2)
        symmetry = round(symmetry, 4)

        profile = self._classify_profile(curvature_deg, symmetry)

        return {
            "midline_length_mm": midline_length_mm,
            "max_width_mm": max_width_mm,
            "curvature_angle_deg": curvature_deg,
            "symmetry_score": symmetry,
            "profile": profile,
        }

    # ── Internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _ensure_binary(mask: np.ndarray) -> np.ndarray:
        """Convert any mask to a clean binary uint8 image."""
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return binary.astype(np.uint8)

    @staticmethod
    def _largest_contour(binary: np.ndarray) -> Optional[np.ndarray]:
        """Return the largest contour by area, or None."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def _align_contour(
        contour: np.ndarray,
        mask: np.ndarray,
        center: Tuple[float, float],
        angle_deg: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate contour and mask so the major axis is horizontal, centred at origin.

        Returns the rotated contour (Nx1x2 int32) and the rotated binary mask.
        """
        cx, cy = center
        h, w = mask.shape[:2]

        # Rotation matrix (rotate by -angle to make major axis horizontal)
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

        # Rotate the mask image
        rotated_mask = cv2.warpAffine(
            mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        # Rotate contour points
        pts = contour.reshape(-1, 2).astype(np.float64)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])  # Nx3
        rotated_pts = (rot_mat @ pts_h.T).T  # Nx2
        aligned_contour = rotated_pts.reshape(-1, 1, 2).astype(np.int32)

        return aligned_contour, rotated_mask

    @staticmethod
    def _sample_midline(
        aligned_contour: np.ndarray,
        aligned_mask: np.ndarray,
        num_samples: int = 50,
    ) -> Tuple[List[Tuple[float, float]], List[float], List[float], float]:
        """
        Sample midline points along the horizontal (major-axis) direction
        of the aligned bean.

        For each vertical slice, find the top and bottom boundary of the bean
        and record the midpoint.

        Returns
        -------
        midline_points : list of (x, y_mid)
        upper_dists    : list of distances from midline to upper boundary
        lower_dists    : list of distances from midline to lower boundary
        span_px        : horizontal span in pixels
        """
        pts = aligned_contour.reshape(-1, 2)
        if len(pts) == 0:
            return [], [], [], 0.0

        x_min = int(pts[:, 0].min())
        x_max = int(pts[:, 0].max())
        span_px = float(max(1, x_max - x_min))

        h, w = aligned_mask.shape[:2]
        step = max(1, int(span_px / num_samples))

        midline_points: List[Tuple[float, float]] = []
        upper_dists: List[float] = []
        lower_dists: List[float] = []

        for x in range(x_min + step, x_max - step + 1, step):
            if x < 0 or x >= w:
                continue

            col = aligned_mask[:, x]
            white_rows = np.where(col > 127)[0]
            if len(white_rows) < 2:
                continue

            y_top = float(white_rows[0])
            y_bot = float(white_rows[-1])
            y_mid = (y_top + y_bot) / 2.0
            half_width_top = y_mid - y_top
            half_width_bot = y_bot - y_mid

            midline_points.append((float(x), y_mid))
            upper_dists.append(half_width_top)
            lower_dists.append(half_width_bot)

        return midline_points, upper_dists, lower_dists, span_px

    @staticmethod
    def _compute_curvature_angle(
        midline_points: List[Tuple[float, float]],
        span_px: float,
    ) -> float:
        """
        Compute the curvature angle of the midline.

        Fits a straight line to the midline points, measures the maximum
        perpendicular deviation, and converts that to an angle:

            curvature_angle = arctan(max_deviation / (span / 2))

        A perfectly straight bean gives 0°.
        """
        if len(midline_points) < 3 or span_px <= 0:
            return 0.0

        xs = np.array([p[0] for p in midline_points], dtype=np.float64)
        ys = np.array([p[1] for p in midline_points], dtype=np.float64)

        # Fit line: y = mx + b
        coeffs = np.polyfit(xs, ys, 1)
        fitted = np.polyval(coeffs, xs)
        deviations = np.abs(ys - fitted)
        max_dev = float(deviations.max())

        half_span = span_px / 2.0
        if half_span <= 0:
            return 0.0

        angle_rad = math.atan2(max_dev, half_span)
        angle_deg = math.degrees(angle_rad)

        return min(angle_deg, 90.0)

    @staticmethod
    def _compute_symmetry(
        upper_dists: List[float],
        lower_dists: List[float],
    ) -> float:
        """
        Compute mirror symmetry about the midline.

        For each sample position, local symmetry is:

            S(i) = 1.0 - |upper(i) - lower(i)| / (upper(i) + lower(i))

        Overall symmetry is the mean of all S(i), clamped to [0, 1].
        """
        if not upper_dists or not lower_dists:
            return 0.0

        n = min(len(upper_dists), len(lower_dists))
        if n == 0:
            return 0.0

        local_scores: List[float] = []
        for i in range(n):
            u = upper_dists[i]
            lo = lower_dists[i]
            total = u + lo
            if total <= 0:
                continue
            diff = abs(u - lo)
            local_scores.append(1.0 - (diff / total))

        if not local_scores:
            return 0.0

        score = float(np.mean(local_scores))
        return max(0.0, min(1.0, score))

    @staticmethod
    def _classify_profile(curvature_deg: float, symmetry: float) -> str:
        """Map curvature angle and symmetry to a named shape profile."""
        if curvature_deg >= _PROFILE_CURVED_MAX_DEG or symmetry < _SYMMETRY_IRREGULAR_THRESHOLD:
            return "IRREGULAR"
        if curvature_deg >= _PROFILE_SLIGHTLY_CURVED_MAX_DEG:
            return "CURVED"
        if curvature_deg >= _PROFILE_STRAIGHT_MAX_DEG:
            return "SLIGHTLY_CURVED"
        return "STRAIGHT"

    @staticmethod
    def _empty_result() -> Dict:
        """Return a zeroed-out result dict when analysis is not possible."""
        return {
            "midline_length_mm": 0.0,
            "max_width_mm": 0.0,
            "curvature_angle_deg": 0.0,
            "symmetry_score": 0.0,
            "profile": "IRREGULAR",
        }


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ShapeAnalyzer — self-test with synthetic bean")
    print("=" * 60)

    # Create a synthetic bean: a slightly rotated ellipse on black canvas
    canvas = np.zeros((300, 400), dtype=np.uint8)
    # Ellipse centred at (200, 150), axes (90, 35), rotated 20°
    cv2.ellipse(canvas, (200, 150), (90, 35), 20, 0, 360, 255, -1)

    # Simulated tray calibration: 400px wide image on a 180mm tray
    sim_px_per_mm_x = 400.0 / 180.0   # ≈ 2.22 px/mm
    sim_px_per_mm_y = 300.0 / 114.0   # ≈ 2.63 px/mm

    analyzer = ShapeAnalyzer()
    result = analyzer.analyze(canvas, sim_px_per_mm_x, sim_px_per_mm_y)

    print(f"  midline_length_mm  : {result['midline_length_mm']}")
    print(f"  max_width_mm       : {result['max_width_mm']}")
    print(f"  curvature_angle_deg: {result['curvature_angle_deg']}")
    print(f"  symmetry_score     : {result['symmetry_score']}")
    print(f"  profile            : {result['profile']}")

    # Basic sanity checks
    assert result["midline_length_mm"] > 0, "Length should be positive"
    assert result["max_width_mm"] > 0, "Width should be positive"
    assert result["midline_length_mm"] > result["max_width_mm"], "Length > width for an ellipse"
    assert 0.0 <= result["symmetry_score"] <= 1.0, "Symmetry must be in [0, 1]"
    assert result["symmetry_score"] > 0.85, f"Ellipse should be highly symmetric, got {result['symmetry_score']}"
    assert result["profile"] in ("STRAIGHT", "SLIGHTLY_CURVED", "CURVED", "IRREGULAR"), \
        f"Invalid profile: {result['profile']}"

    print("\n  All assertions PASSED.")

    # Test with an asymmetric / irregular shape
    canvas2 = np.zeros((300, 400), dtype=np.uint8)
    pts = np.array([[100, 150], [200, 80], [350, 130], [300, 200], [150, 220]], dtype=np.int32)
    cv2.fillPoly(canvas2, [pts], 255)

    result2 = analyzer.analyze(canvas2, sim_px_per_mm_x, sim_px_per_mm_y)
    print(f"\n  Irregular shape test:")
    print(f"    symmetry_score : {result2['symmetry_score']}")
    print(f"    curvature_deg  : {result2['curvature_angle_deg']}")
    print(f"    profile        : {result2['profile']}")

    # Test empty mask
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    result3 = analyzer.analyze(empty_mask, sim_px_per_mm_x, sim_px_per_mm_y)
    assert result3["profile"] == "IRREGULAR", "Empty mask should be IRREGULAR"
    assert result3["midline_length_mm"] == 0.0, "Empty mask should have zero length"
    print(f"\n  Empty mask test: profile={result3['profile']} — PASSED")

    print("\n" + "=" * 60)
    print("All self-tests completed successfully.")
    print("=" * 60)
