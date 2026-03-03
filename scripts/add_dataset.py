"""
Add a new YOLO-format dataset to the consolidated folder.

Usage:
    python scripts/add_dataset.py <dataset_folder> [--split-ratio 0.8 0.1 0.1]

The dataset_folder should contain:
  images/  (jpg/jpeg/png files)
  labels/  (corresponding .txt YOLO labels)

If the dataset is pre-split (has train/val/test subfolders), those splits are preserved.
Otherwise, images are randomly split into train/val/test per --split-ratio.
"""
import argparse
import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONSOLIDATED = ROOT / "datasets" / "consolidated"
UNANNOTATED = ROOT / "datasets" / "unannotated"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def safe_copy(src: Path, dst_dir: Path):
    """Copy file, renaming on conflict. Return the new filename stem."""
    dst = dst_dir / src.name
    stem = src.stem
    ext = src.suffix
    i = 1
    while dst.exists():
        stem = f"{src.stem}_{i}"
        dst = dst_dir / f"{stem}{ext}"
        i += 1
    shutil.copy2(src, dst)
    return stem


def add_split(img_dir: Path, lbl_dir: Path, target_split: str):
    """Add one split (images+labels) from source to consolidated."""
    if not img_dir.is_dir():
        return 0, 0

    target_img = CONSOLIDATED / "images" / target_split
    target_lbl = CONSOLIDATED / "labels" / target_split
    target_img.mkdir(parents=True, exist_ok=True)
    target_lbl.mkdir(parents=True, exist_ok=True)
    UNANNOTATED.mkdir(parents=True, exist_ok=True)

    added = 0
    skipped = 0
    images = [f for f in img_dir.iterdir()
              if f.is_file() and f.suffix.lower() in IMAGE_EXTS]

    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            new_stem = safe_copy(img, target_img)
            shutil.copy2(lbl, target_lbl / (new_stem + ".txt"))
            added += 1
        else:
            safe_copy(img, UNANNOTATED)
            skipped += 1

    return added, skipped


def detect_structure(dataset_path: Path):
    """Detect if the dataset is pre-split or flat."""
    # Check for pre-split structure
    for split_name in ("train", "val", "valid", "test"):
        if (dataset_path / split_name / "images").is_dir():
            return "pre_split_nested"
        if (dataset_path / "images" / split_name).is_dir():
            return "pre_split_flat"

    # Flat structure: images/ + labels/
    if (dataset_path / "images").is_dir() and (dataset_path / "labels").is_dir():
        return "flat"

    # Direct images and labels in root
    has_images = any(f.suffix.lower() in IMAGE_EXTS
                     for f in dataset_path.iterdir() if f.is_file())
    if has_images:
        return "root_flat"

    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Add a new dataset to the consolidated folder")
    parser.add_argument("dataset", help="Path to the new dataset folder")
    parser.add_argument("--split-ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Train/val/test split ratio for flat datasets (default: 0.8 0.1 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.is_dir():
        print(f"❌ Not a directory: {dataset_path}")
        return

    structure = detect_structure(dataset_path)
    print(f"\n📂 Dataset: {dataset_path}")
    print(f"📐 Structure detected: {structure}")

    total_added = 0
    total_skipped = 0

    if structure == "pre_split_nested":
        # train/images, train/labels, val/images, val/labels, test/images, test/labels
        for split in ("train", "val", "valid", "test"):
            target = "val" if split == "valid" else split
            img_dir = dataset_path / split / "images"
            lbl_dir = dataset_path / split / "labels"
            if img_dir.is_dir():
                a, s = add_split(img_dir, lbl_dir, target)
                total_added += a
                total_skipped += s
                print(f"  ✅ {split} → {target}: +{a} images ({s} skipped)")

    elif structure == "pre_split_flat":
        # images/train, images/val, labels/train, labels/val
        for split in ("train", "val", "valid", "test"):
            target = "val" if split == "valid" else split
            img_dir = dataset_path / "images" / split
            lbl_dir = dataset_path / "labels" / split
            if img_dir.is_dir():
                a, s = add_split(img_dir, lbl_dir, target)
                total_added += a
                total_skipped += s
                print(f"  ✅ {split} → {target}: +{a} images ({s} skipped)")

    elif structure in ("flat", "root_flat"):
        # Random split
        random.seed(args.seed)
        img_dir = dataset_path / "images" if structure == "flat" else dataset_path
        lbl_dir = dataset_path / "labels" if structure == "flat" else dataset_path

        images = sorted([f for f in img_dir.iterdir()
                         if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
        random.shuffle(images)

        n = len(images)
        n_train = int(n * args.split_ratio[0])
        n_val = int(n * args.split_ratio[1])

        splits = {}
        splits["train"] = images[:n_train]
        splits["val"] = images[n_train:n_train + n_val]
        splits["test"] = images[n_train + n_val:]

        for split, imgs in splits.items():
            target_img = CONSOLIDATED / "images" / split
            target_lbl = CONSOLIDATED / "labels" / split
            target_img.mkdir(parents=True, exist_ok=True)
            target_lbl.mkdir(parents=True, exist_ok=True)
            UNANNOTATED.mkdir(parents=True, exist_ok=True)

            added = 0
            skipped = 0
            for img in imgs:
                lbl = lbl_dir / (img.stem + ".txt")
                if lbl.exists() and lbl.stat().st_size > 0:
                    new_stem = safe_copy(img, target_img)
                    shutil.copy2(lbl, target_lbl / (new_stem + ".txt"))
                    added += 1
                else:
                    safe_copy(img, UNANNOTATED)
                    skipped += 1
            total_added += added
            total_skipped += skipped
            print(f"  ✅ → {split}: +{added} images ({skipped} skipped)")
    else:
        print(f"  ❌ Could not detect dataset structure. Expected:")
        print(f"     - images/ + labels/ folders, or")
        print(f"     - train/val/test subfolders with images/labels")
        return

    print(f"\n  📊 Total added: {total_added}, Unannotated skipped: {total_skipped}")

    # Show updated totals
    for split in ("train", "val", "test"):
        count = len(list((CONSOLIDATED / "images" / split).iterdir()))
        print(f"  {split:<6s} → {count} images total")
    print()


if __name__ == "__main__":
    main()
