# How to Run & Deploy the Coffee Bean Analyzer

This guide explains how to run the Coffee Bean Analyzer on your machine, package it for delivery, and install it on other laptops or devices.

---

## 1. Quick Start: Running on This Machine

If you are running the project from the source folder on a system with Python installed, you can launch the application in two ways:

### Method A: Double-Click the Executable Launcher (Recommended)
1. Double-click **`CoffeeBeanAnalyzer.exe`** in the root directory.
2. The bootstrapper launcher will run in the terminal, check your dependencies, create/verify the virtual environment (`.venv`), install missing packages, and start the local Flask web server automatically.
3. It will launch Google Chrome in app-window mode pointing to `http://localhost:5000`.

### Method B: Start via Terminal (Manual)
1. Open PowerShell or Command Prompt in the project folder.
2. Activate your virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
3. Start the Flask application:
   ```bash
   python website/app.py
   ```
4. Open your browser and navigate to `http://localhost:5000`.

---

## 2. Deploying on Other Systems

The bootstrap executable (`CoffeeBeanAnalyzer.exe`) is designed to be lightweight. It does **not** bundle the large deep learning packages (like PyTorch and CUDA) inside the `.exe` binary. Instead, it builds the environment on the host machine when run for the first time.

To distribute the app to another system, follow these steps:

### Step 1: Rebuild the Packaging Zip
Ensure all code changes and dependencies are packaged:
```bash
python _package.py
```
This builds a zip file named `coffee_bean_classifier_<date>.zip` containing all application scripts, configs, and reports, excluding heavy virtual environments and cache directories.

### Step 2: Transfer and Install on the New Device
1.  **Transfer the Zip:** Share the generated `.zip` archive with the target user.
2.  **Extract the Files:** Have the user extract the zip archive into a folder on their local drive.
3.  **Run the Bootstrapper:** Double-click **`CoffeeBeanAnalyzer.exe`** inside the extracted folder.
    *   *First-time setup:* The launcher will automatically detect if Python 3.10+ is installed on the machine, create a local `.venv` virtual environment, and download and install the required dependencies (including Torch, OpenCV, and SAM 2).
    *   Once setup is complete, it will launch the Flask backend server and automatically open the application interface in Chrome/default browser.

*Note: The first launch on a new system requires an internet connection and will take a few minutes to download the machine learning dependencies.*

---

## 3. System Requirements

*   **Operating System:** Windows 10 or 11 (64-bit).
*   **Python:** Python 3.10, 3.11, or 3.12 installed and added to the System Path (`PATH`).
*   **Hardware (CUDA GPU):** NVIDIA GPU with >= 4 GB VRAM (e.g. GTX 1650 or higher) is highly recommended for real-time SAM 2 segmentation (~78ms decoder speed).
*   **Hardware (CPU Fallback):** Laptops without an NVIDIA GPU can run the app, but SAM 2 will run on the CPU (taking 5-15 seconds per image) or the pipeline will automatically fall back to fast OpenCV Otsu-thresholding silhouette analysis.
