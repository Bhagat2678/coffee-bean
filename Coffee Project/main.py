import os
from src.detector import count_beans_opencv

# Define folder paths
RAW_DIR = "data/raw"
OUTPUT_DIR = "data/output"

def get_valid_image_path():
    """
    Asks user for filename and checks if it exists.
    """
    while True:
        print("\n" + "="*40)
        print(f"📂 Available images in '{RAW_DIR}':")
        
        # List all files in the data/raw folder so you know what to type
        try:
            files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not files:
                print("   (No images found! Please add an image first.)")
                return None
            for f in files:
                print(f"   - {f}")
        except FileNotFoundError:
            print(f"   ❌ Error: Folder '{RAW_DIR}' does not exist.")
            return None

        print("="*40)
        filename = input("👉 Enter the image name (e.g., test_beans.jpg): ").strip()
        
        # Construct full path
        full_path = os.path.join(RAW_DIR, filename)
        
        if os.path.exists(full_path):
            return filename
        else:
            print(f"❌ Error: File '{filename}' not found. Try again.")

def main():
    # 1. Ask user for the image
    filename = get_valid_image_path()
    if not filename:
        return

    # 2. Define input and output paths
    input_path = os.path.join(RAW_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, f"analyzed_{filename}")

    print(f"\n☕ Processing '{filename}'...")
    
    # 3. Run the detector
    count = count_beans_opencv(input_path, output_path)
    
    # 4. Show results
    print("-" * 30)
    print(f"✅ Success!")
    print(f"🫘  Total Beans Counted: {count}")
    print(f"🖼️  Result image saved to: {output_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()