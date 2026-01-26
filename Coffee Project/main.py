import os
import time
from pathlib import Path
import cv2
from src.detector import count_beans_opencv
from src.database import get_db_manager
from src.config import RAW_DIR, OUTPUT_DIR

def get_valid_image_path():
    """Asks user for filename and checks if it exists."""
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

def show_menu():
    """Display main menu and get user choice."""
    print("\n" + "="*50)
    print("☕ COFFEE BEAN ANALYSIS SYSTEM")
    print("="*50)
    print("1. Analyze a new image")
    print("2. View all analyses")
    print("3. View statistics")
    print("4. View history for specific image")
    print("5. Delete an analysis record")
    print("6. Exit")
    print("="*50)
    choice = input("👉 Enter your choice (1-6): ").strip()
    return choice

def analyze_image():
    """Analyze a coffee bean image and save results to database."""
    # 1. Ask user for the image
    filename = get_valid_image_path()
    if not filename:
        return

    # 2. Define input and output paths
    input_path = os.path.join(RAW_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, f"analyzed_{filename}")

    print(f"\n☕ Processing '{filename}'...")
    
    # 3. Record start time for processing duration
    start_time = time.time()
    
    # 4. Run the detector
    count = count_beans_opencv(input_path, output_path)
    processing_time = time.time() - start_time
    
    # 5. Get image dimensions
    img = cv2.imread(input_path)
    height, width = img.shape[:2] if img is not None else (None, None)
    
    # 6. Save results to database
    db_manager = get_db_manager()
    analysis = db_manager.add_analysis(
        image_name=filename,
        image_path=os.path.abspath(input_path),
        output_path=os.path.abspath(output_path),
        bean_count=count,
        width=width,
        height=height,
        processing_time=processing_time,
        notes="Automatic analysis via main.py"
    )
    
    # 7. Show results
    print("-" * 50)
    print(f"✅ Success!")
    print(f"🫘  Total Beans Counted: {count}")
    print(f"📐 Image Dimensions: {width}x{height}")
    print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
    print(f"🖼️  Result image saved to: {output_path}")
    print(f"💾 Database ID: {analysis.id}")
    print("-" * 50)

def view_all_analyses():
    """Display all analyses from the database."""
    db_manager = get_db_manager()
    analyses = db_manager.get_all_analyses()
    
    if not analyses:
        print("\n⚠️  No analyses found in database.")
        return
    
    print("\n" + "="*80)
    print("📊 ALL ANALYSES")
    print("="*80)
    print(f"{'ID':<5} {'Image':<25} {'Beans':<8} {'Dimensions':<15} {'Date':<20}")
    print("-"*80)
    
    for analysis in analyses:
        dims = f"{analysis.image_width}x{analysis.image_height}" if analysis.image_width else "N/A"
        date_str = analysis.created_at.strftime("%Y-%m-%d %H:%M") if analysis.created_at else "N/A"
        print(f"{analysis.id:<5} {analysis.image_name:<25} {analysis.bean_count:<8} {dims:<15} {date_str:<20}")
    
    print("="*80)

def view_statistics():
    """Display analysis statistics."""
    db_manager = get_db_manager()
    stats = db_manager.get_statistics()
    
    print("\n" + "="*50)
    print("📈 STATISTICS")
    print("="*50)
    print(f"Total Analyses: {stats['total_analyses']}")
    print(f"Unique Images Processed: {stats['total_images_processed']}")
    print(f"Average Bean Count: {stats['average_bean_count']:.1f}")
    print(f"Min Bean Count: {stats['min_bean_count']}")
    print(f"Max Bean Count: {stats['max_bean_count']}")
    print("="*50)

def view_image_history():
    """View analysis history for a specific image."""
    filename = input("👉 Enter image name (e.g., test_beans.jpg): ").strip()
    
    db_manager = get_db_manager()
    analyses = db_manager.get_analyses_by_image(filename)
    
    if not analyses:
        print(f"\n⚠️  No analyses found for '{filename}'.")
        return
    
    print("\n" + "="*80)
    print(f"📋 HISTORY FOR '{filename}'")
    print("="*80)
    print(f"{'ID':<5} {'Beans':<8} {'Dimensions':<15} {'Processing Time':<18} {'Date':<20}")
    print("-"*80)
    
    for analysis in analyses:
        dims = f"{analysis.image_width}x{analysis.image_height}" if analysis.image_width else "N/A"
        proc_time = f"{analysis.processing_time:.2f}s" if analysis.processing_time else "N/A"
        date_str = analysis.created_at.strftime("%Y-%m-%d %H:%M") if analysis.created_at else "N/A"
        print(f"{analysis.id:<5} {analysis.bean_count:<8} {dims:<15} {proc_time:<18} {date_str:<20}")
    
    print("="*80)

def delete_analysis_record():
    """Delete an analysis record from the database."""
    try:
        analysis_id = int(input("👉 Enter analysis ID to delete: ").strip())
        db_manager = get_db_manager()
        success = db_manager.delete_analysis(analysis_id)
        if not success:
            print(f"⚠️  Analysis {analysis_id} not found.")
    except ValueError:
        print("❌ Invalid ID. Please enter a number.")

def main():
    """Main application loop."""
    try:
        while True:
            choice = show_menu()
            
            if choice == "1":
                analyze_image()
            elif choice == "2":
                view_all_analyses()
            elif choice == "3":
                view_statistics()
            elif choice == "4":
                view_image_history()
            elif choice == "5":
                delete_analysis_record()
            elif choice == "6":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
