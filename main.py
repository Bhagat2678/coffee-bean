import os
import sys
from src.detector import ObjectDetector
from src.analyzer import BeanAnalyzer

# Folder paths
RAW_DIR = "data/raw"
OUTPUT_DIR = "data/output"

# Model configuration
MODEL_PATH = "models/best.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.01


def get_valid_image_path(auto_select=None):
    """
    Gets image filename. If auto_select is provided, uses it directly.
    Otherwise asks user for filename.
    """
    if auto_select:
        full_path = os.path.join(RAW_DIR, auto_select)
        if os.path.exists(full_path):
            return auto_select
        else:
            print(f"❌ Error: File '{auto_select}' not found.")
            return None

    # Interactive mode
    while True:
        print("\n" + "=" * 40)
        print(f"📂 Available images in '{RAW_DIR}':")

        try:
            files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not files:
                print("   (No images found! Please add an image first.)")
                return None
            for i, f in enumerate(files, 1):
                print(f"   {i}. {f}")
        except FileNotFoundError:
            print(f"   ❌ Error: Folder '{RAW_DIR}' does not exist.")
            return None

        print("=" * 40)
        filename = input("👉 Enter the image name (or just press Enter for first): ").strip()

        if not filename and files:
            return files[0]

        full_path = os.path.join(RAW_DIR, filename)
        if os.path.exists(full_path):
            return filename
        else:
            print(f"❌ Error: File '{filename}' not found. Try again.")


def main():
    auto_image = None
    if len(sys.argv) > 1:
        auto_image = sys.argv[1]
        print(f"📸 Processing: {auto_image}")

    filename = get_valid_image_path(auto_select=auto_image)
    if not filename:
        return

    input_path = os.path.join(RAW_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, f"detected_{filename}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n🤖 Loading model: {MODEL_PATH}")
    detector = ObjectDetector(MODEL_PATH)

    print(f"🔍 Processing '{filename}'...")
    print(f"🔄 Analyzing color, size, and bean count...\n")

    analyzer = BeanAnalyzer(detector)
    results = analyzer.analyze_image(input_path, CONFIDENCE_THRESHOLD, output_path)

    # Show results
    print("=" * 50)
    print(f"✅ Analysis Complete!")
    print("=" * 50)

    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return

    summary = results['summary']
    print(f"📊 Total Beans Detected: {summary['total_count']}")

    if summary['color_distribution']:
        print(f"\n🎨 BEAN COLORS:")
        for color, count in sorted(summary['color_distribution'].items(), key=lambda x: -x[1]):
            pct = (count / summary['total_count'] * 100) if summary['total_count'] > 0 else 0
            print(f"   • {color}: {count} ({pct:.1f}%)")

    if summary['size_distribution']:
        print(f"\n📐 BEAN SIZES:")
        for size_class in ["Tiny", "Small", "Medium", "Large", "Very Large"]:
            if size_class in summary['size_distribution']:
                count = summary['size_distribution'][size_class]
                pct = (count / summary['total_count'] * 100) if summary['total_count'] > 0 else 0
                print(f"   • {size_class}: {count} ({pct:.1f}%)")

    print(f"\n🖼️  Annotated image saved to: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
