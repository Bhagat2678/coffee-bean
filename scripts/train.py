"""
Train YOLOv8 on the coffee beans dataset.

Usage:
    python scripts/train.py [--epochs 150] [--batch 16] [--imgsz 640]
"""
import argparse
import os
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = ROOT / "datasets" / "consolidated" / "dataset.yaml"
DEFAULT_MODEL = str(ROOT / "models" / "best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on consolidated coffee beans dataset")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model path (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs (default: 150)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", default="0", help="Device: '0' for GPU, 'cpu' for CPU (default: 0)")
    parser.add_argument("--name", default="consolidated_v1", help="Run name (default: consolidated_v1)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    if not DATASET_YAML.exists():
        print(f"❌ Dataset config not found: {DATASET_YAML}")
        return

    print("=" * 55)
    print("  Coffee Beans — Model Training")
    print("=" * 55)
    print(f"  📦 Dataset:  {DATASET_YAML}")
    print(f"  🤖 Model:    {args.model}")
    print(f"  🔁 Epochs:   {args.epochs}")
    print(f"  📐 Img size: {args.imgsz}")
    print(f"  📦 Batch:    {args.batch}")
    print(f"  💻 Device:   {'GPU' if args.device != 'cpu' else 'CPU'}")
    print(f"  📁 Run name: {args.name}")
    print("=" * 55)

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        # Optimization
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        # Save
        save=True,
        save_period=25,
        plots=True,
        verbose=True,
    )

    # Print final results location
    best_model = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    print("\n" + "=" * 55)
    print("  ✅ Training Complete!")
    print(f"  📁 Best model: {best_model}")
    print("=" * 55)

    # Validate on test set
    print("\n🔍 Running validation on test set...")
    val_results = model.val(data=str(DATASET_YAML), split="test")
    print(f"  mAP50:    {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
