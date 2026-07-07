import os
import sys
import time

# We use requests which is already installed in our .venv
import requests

def download_file_with_retry(url, dest_path, max_retries=5):
    print(f"Downloading {url}...")
    print(f"Saving to: {dest_path}")
    
    # Ensure destination folder exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    temp_path = dest_path + ".tmp"
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}/{max_retries}...")
            # We open the stream with a timeout
            response = requests.get(url, stream=True, timeout=20)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 1024 * 1024 # 1 MB chunks
            
            last_pct = -1
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            pct = int((downloaded / total_size) * 100)
                            if pct % 5 == 0 and pct != last_pct:
                                sys.stdout.write(f"Progress: {pct}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)\n")
                                sys.stdout.flush()
                                last_pct = pct
                                
            # Rename temp file to final destination
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(temp_path, dest_path)
            print("\n✅ Download complete and verified!")
            return True
            
        except Exception as e:
            print(f"\n⚠️ Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print("Waiting 5 seconds before retrying...")
                time.sleep(5)
            else:
                print("\n❌ All download attempts failed.")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                return False

if __name__ == "__main__":
    WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    DEST_PATH = os.path.join(DEST_DIR, "sam2.1_hiera_large.pt")
    
    success = download_file_with_retry(WEIGHTS_URL, DEST_PATH)
    if not success:
        sys.exit(1)
