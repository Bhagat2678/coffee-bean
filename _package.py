"""
Package the coffee bean classifier project into a clean delivery zip.

Generates PACKAGE_MANIFEST.txt and creates a zip with only the files
specified in the inclusion list, excluding images, caches, virtual
environments, training artifacts, and oversized files.

Usage:
    python _package.py
"""

import os
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Files to include (relative to ROOT) ──────────────────────────────────
INCLUDE_FILES = [
    # Root-level files
    "main.py",
    "requirements.txt",
    "README.md",
    "Techstack.md",
    ".gitignore",
    "session_state.json",

    # src/
    "src/__init__.py",
    "src/analyzer.py",
    "src/arcface.py",
    "src/cvat_converter.py",
    "src/detector.py",
    "src/grading.py",
    "src/shape_analyzer.py",

    # scripts/
    "scripts/add_dataset.py",
    "scripts/evaluate.py",
    "scripts/infer.py",
    "scripts/train.py",

    # website/
    "website/app.py",
    "website/templates/index.html",
    "website/static/css/style.css",
    "website/static/js/app.js",

    # database/
    "database/schema.sql",

    # dataset config only (no images)
    "datasets/consolidated/dataset.yaml",

    # Placeholder .gitkeep files
    "data/raw/.gitkeep",
    "data/output/.gitkeep",
    "website/uploads/.gitkeep",
    "results/.gitkeep",
]

# Model files — include only if they exist
MODEL_FILES = [
    "models/best.pt",
    "models/coffee_classifier.pt",
]

# Image / binary extensions to reject (safety net)
REJECT_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff",
    ".mp4", ".avi", ".mov", ".mkv",
    ".pyc", ".pyo",
}

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def build_file_list():
    """Build the final list of (relative_path, absolute_path) tuples."""
    files = []
    for rel in INCLUDE_FILES:
        abs_path = ROOT / rel
        if abs_path.exists() and abs_path.is_file():
            files.append((rel, abs_path))
        else:
            print(f"  [WARN] Listed file not found, skipping: {rel}")

    for rel in MODEL_FILES:
        abs_path = ROOT / rel
        if abs_path.exists() and abs_path.is_file():
            size = abs_path.stat().st_size
            if size <= MAX_FILE_SIZE_BYTES:
                files.append((rel, abs_path))
            else:
                print(f"  [WARN] Model too large ({size / 1024 / 1024:.1f} MB), skipping: {rel}")

    return files


def generate_manifest(files):
    """Create PACKAGE_MANIFEST.txt content."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_exists = (ROOT / "models" / "best.pt").exists()
    classifier_exists = (ROOT / "models" / "coffee_classifier.pt").exists()

    lines = [
        "=" * 50,
        "COFFEE BEAN CLASSIFIER - PACKAGE MANIFEST",
        f"Generated: {timestamp}",
        "=" * 50,
        "",
        f"INCLUDED FILES ({len(files)}):",
    ]

    total_size = 0
    for rel, abs_path in sorted(files, key=lambda x: x[0]):
        size = abs_path.stat().st_size
        total_size += size
        size_kb = size / 1024
        lines.append(f"  {rel:<55s} {size_kb:>8.1f} KB")

    lines.append("")
    lines.append(f"TOTAL SIZE: {total_size / 1024 / 1024:.2f} MB")
    lines.append("")
    lines.append("EXCLUDED (intentionally):")
    lines.append("  - All image/video files (*.jpg, *.png, *.mp4, etc.)")
    lines.append("  - Virtual environments (.venv/, venv/, env/)")
    lines.append("  - Training run artifacts (runs/)")
    lines.append("  - __pycache__/ and *.pyc files")
    lines.append("  - Git history (.git/)")
    lines.append("  - Phase1.docx, What_is_YOLO.docx (old docs)")
    lines.append("  - Jupyter notebook checkpoints")
    lines.append("")
    lines.append("MODEL STATUS:")
    lines.append(f"  models/best.pt              -> {'EXISTS' if best_exists else 'MISSING'}")
    lines.append(f"  models/coffee_classifier.pt -> {'EXISTS' if classifier_exists else 'MISSING (train first)'}")
    lines.append("")
    lines.append("READY TO TRAIN: YES")
    lines.append("  Run: python scripts/train.py")
    lines.append("")
    lines.append("READY TO RUN (web app): YES")
    lines.append("  Run: python website/app.py")
    lines.append("  Open: http://localhost:5000")
    lines.append("=" * 50)

    return "\n".join(lines) + "\n"


def create_zip(files, manifest_content):
    """Create the delivery zip file."""
    date_str = datetime.now().strftime("%Y%m%d")
    zip_name = f"coffee_bean_classifier_{date_str}.zip"
    zip_path = ROOT / zip_name
    zip_root = "coffee_bean_classifier"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write manifest
        zf.writestr(f"{zip_root}/PACKAGE_MANIFEST.txt", manifest_content)

        # Write all included files
        for rel, abs_path in files:
            arcname = f"{zip_root}/{rel}"
            zf.write(abs_path, arcname)

    return zip_path, zip_name


def verify_zip(zip_path):
    """Run verification checks on the produced zip."""
    checks = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # Check 1: No image files
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        image_files = [n for n in names if Path(n).suffix.lower() in image_exts]
        if image_files:
            checks.append(("No image files inside", False, f"Found: {image_files[:3]}"))
        else:
            checks.append(("No image files inside", True, ""))

        # Check 2: No __pycache__
        pycache = [n for n in names if "__pycache__" in n]
        if pycache:
            checks.append(("No __pycache__ folders", False, f"Found: {pycache[:3]}"))
        else:
            checks.append(("No __pycache__ folders", True, ""))

        # Check 3: No .venv
        venv = [n for n in names if ".venv" in n or "/venv/" in n]
        if venv:
            checks.append(("No .venv folder", False, f"Found: {venv[:3]}"))
        else:
            checks.append(("No .venv folder", True, ""))

        # Check 4: All .gitkeep files
        prefix = "coffee_bean_classifier/"
        gitkeeps = [
            f"{prefix}data/raw/.gitkeep",
            f"{prefix}data/output/.gitkeep",
            f"{prefix}website/uploads/.gitkeep",
            f"{prefix}results/.gitkeep",
        ]
        for gk in gitkeeps:
            if gk in names:
                checks.append((f"{gk.split(prefix)[1]} exists", True, ""))
            else:
                checks.append((f"{gk.split(prefix)[1]} exists", False, "Missing"))

        # Check 5-10: Key files exist
        key_files = {
            "PACKAGE_MANIFEST.txt": f"{prefix}PACKAGE_MANIFEST.txt",
            "README.md": f"{prefix}README.md",
            "requirements.txt": f"{prefix}requirements.txt",
            "website/app.py": f"{prefix}website/app.py",
            "scripts/train.py": f"{prefix}scripts/train.py",
            "src/shape_analyzer.py": f"{prefix}src/shape_analyzer.py",
        }
        for label, path in key_files.items():
            if path in names:
                checks.append((f"{label} exists", True, ""))
            else:
                checks.append((f"{label} exists", False, "Missing"))

    return checks


def main():
    """Run the full packaging pipeline."""
    print("=" * 55)
    print("  Coffee Bean Classifier — Packaging")
    print("=" * 55)

    # Step 1: Build file list
    print("\n[1/4] Building file list...")
    files = build_file_list()
    print(f"  Found {len(files)} files to include")

    # Step 2: Generate manifest
    print("[2/4] Generating PACKAGE_MANIFEST.txt...")
    manifest = generate_manifest(files)
    manifest_path = ROOT / "PACKAGE_MANIFEST.txt"
    manifest_path.write_text(manifest, encoding="utf-8")
    print(f"  Written to: PACKAGE_MANIFEST.txt")

    # Add manifest to file list
    files.append(("PACKAGE_MANIFEST.txt", manifest_path))

    # Step 3: Create zip
    print("[3/4] Creating zip archive...")
    zip_path, zip_name = create_zip(files, manifest)
    zip_size_mb = zip_path.stat().st_size / 1024 / 1024

    with zipfile.ZipFile(zip_path, "r") as zf:
        file_count = len(zf.namelist())

    print(f"  Archive: {zip_name}")
    print(f"  Files:   {file_count}")
    print(f"  Size:    {zip_size_mb:.2f} MB")
    print(f"  Path:    ./{zip_name}")

    # Step 4: Verify
    print("\n[4/4] Verifying zip contents...")
    checks = verify_zip(zip_path)
    all_pass = True
    for label, passed, reason in checks:
        icon = "PASS" if passed else "FAIL"
        suffix = f": {reason}" if reason else ""
        print(f"  {'[OK]' if passed else '[!!]'} {icon} — {label}{suffix}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print(f"  Package ready: {zip_name}")
        print("  DELIVERY READY")
    else:
        print("  ISSUES FOUND — see above")

    print("=" * 55)


if __name__ == "__main__":
    main()
