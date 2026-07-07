import os
import sys
import subprocess
import time
import socket
import webbrowser
import winreg
import shutil
import glob

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message, status="info"):
    if status == "info":
        print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {message}")
    elif status == "success":
        print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {message}")
    elif status == "warning":
        print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {message}")
    elif status == "error":
        print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")
    else:
        print(message)

def get_project_root():
    """Finds the directory containing requirements.txt, checking current or parent directories."""
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
    else:
        exe_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if requirements.txt is in the exe directory
    if os.path.exists(os.path.join(exe_dir, "requirements.txt")):
        return exe_dir
    # Check if requirements.txt is in parent directory (useful when run from dist/)
    parent_dir = os.path.dirname(exe_dir)
    if os.path.exists(os.path.join(parent_dir, "requirements.txt")):
        return parent_dir
    return exe_dir

def find_system_python():
    """Finds a system Python interpreter that can run commands or create venv."""
    if not getattr(sys, 'frozen', False):
        return sys.executable

    python_path = shutil.which("python")
    if python_path:
        try:
            res = subprocess.run([python_path, "--version"], capture_output=True, text=True)
            if res.returncode == 0:
                return python_path
        except Exception:
            pass

    # Try Windows registry
    try:
        import winreg
        for hkey in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(hkey, r"SOFTWARE\Python\PythonCore") as key:
                    i = 0
                    while True:
                        try:
                            ver_name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, rf"{ver_name}\InstallPath") as subkey:
                                val, _ = winreg.QueryValueEx(subkey, "")
                                if val:
                                    py_exe = os.path.join(val, "python.exe")
                                    if os.path.exists(py_exe):
                                        return py_exe
                        except OSError:
                            break
                        i += 1
            except OSError:
                pass
    except ImportError:
        pass

    # Common installation directories on Windows
    common_dirs = [
        os.path.expandvars(r"%LocalAppData%\Programs\Python\Python*"),
        os.path.expandvars(r"%ProgramFiles%\Python*"),
    ]
    for path_pattern in common_dirs:
        for folder in glob.glob(path_pattern):
            py_exe = os.path.join(folder, "python.exe")
            if os.path.exists(py_exe):
                return py_exe

    return "python"

def get_venv_python():
    """Returns the path to the virtual environment python interpreter."""
    if sys.platform == "win32":
        return os.path.abspath(os.path.join(".venv", "Scripts", "python.exe"))
    return os.path.abspath(os.path.join(".venv", "bin", "python"))

def setup_virtual_environment():
    """Checks if virtual environment exists, if not creates it."""
    venv_dir = os.path.abspath(".venv")
    venv_python = get_venv_python()

    if not os.path.exists(venv_dir) or not os.path.exists(venv_python):
        print_status("Virtual environment (.venv) not found. Creating it...", "warning")
        system_python = find_system_python()
        print_status(f"Using system python: {system_python}", "info")
        try:
            # We run system python to create venv
            subprocess.run([system_python, "-m", "venv", ".venv"], check=True)
            print_status("Virtual environment created successfully.", "success")
        except subprocess.CalledProcessError as e:
            print_status(f"Failed to create virtual environment: {e}", "error")
            print("Please ensure Python is installed and added to PATH.")
            input("Press Enter to exit...")
            sys.exit(1)
    else:
        print_status("Virtual environment verified.", "success")

def check_dependencies():
    """Checks if all dependencies in requirements.txt are installed in the venv."""
    venv_python = get_venv_python()
    req_file = "requirements.txt"

    if not os.path.exists(req_file):
        print_status(f"'{req_file}' not found. Skipping dependency check.", "warning")
        return True

    # Read requirements
    with open(req_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    # Map requirement name to Python module name for checking
    # E.g. opencv-python -> cv2, python-dotenv -> dotenv
    import_mapping = {
        "opencv-python": "cv2",
        "python-dotenv": "dotenv",
        "ultralytics": "ultralytics",
        "scipy": "scipy",
        "numpy": "numpy",
        "torch": "torch",
        "torchvision": "torchvision",
        "flask": "flask",
        "werkzeug": "werkzeug",
        "hydra-core": "hydra"
    }

    modules_to_check = []
    for req in requirements:
        # Get package name (before ==, >=, etc.)
        pkg_name = req.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip().lower()
        if not pkg_name:
            continue
        module_name = import_mapping.get(pkg_name, pkg_name)
        modules_to_check.append(module_name)

    # Prepare python statement to check imports
    import_stmt = "; ".join([f"import {mod}" for mod in modules_to_check])
    check_code = f"try:\n    {import_stmt}\n    print('OK')\nexcept Exception as e:\n    print('ERROR:', e)\n    exit(1)"

    print_status("Checking dependencies status...", "info")
    try:
        res = subprocess.run(
            [venv_python, "-c", check_code],
            capture_output=True,
            text=True
        )
        if res.returncode == 0 and "OK" in res.stdout:
            print_status("All dependencies are already installed.", "success")
            return True
        else:
            print_status("Missing dependencies detected.", "warning")
            return False
    except Exception as e:
        print_status(f"Could not verify dependencies: {e}", "warning")
        return False

def install_dependencies():
    """Installs dependencies from requirements.txt into the venv."""
    venv_python = get_venv_python()
    print_status("Installing/updating dependencies. This may take a few minutes...", "info")
    try:
        # Upgrade pip first
        subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        # Install requirements
        subprocess.run([venv_python, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print_status("Dependencies installed successfully.", "success")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to install dependencies: {e}", "error")
        print("\nPlease check your internet connection and try running again.")
        input("Press Enter to exit...")
        sys.exit(1)

def find_chrome():
    """Finds Chrome executable path on Windows."""
    if sys.platform != "win32":
        return None
        
    for key_path in [
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe",
        r"SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe"
    ]:
        for hkey in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hkey, key_path) as key:
                    val, _ = winreg.QueryValueEx(key, "")
                    if val and os.path.exists(val):
                        return val
            except OSError:
                pass
                
    # Fallback paths
    paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def is_port_open(port=5000):
    """Checks if a port is open and listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def wait_for_server(port=5000, timeout=15):
    """Waits until the server is listening on the specified port."""
    print_status("Waiting for web server to start...", "info")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_open(port):
            return True
        time.sleep(0.2)
    return False

def main():
    # Detect and set CWD to project root
    root_dir = get_project_root()
    os.chdir(root_dir)

    # Print welcome ASCII art
    print(f"""{Colors.HEADER}{Colors.BOLD}
============================================================
      COFFEE BEAN DEFECT ANALYZER LAUNCHER
============================================================{Colors.ENDC}""")
    
    # Ensure Windows console supports ANSI colors
    if sys.platform == "win32":
        os.system("")

    # Step 1: Set up Virtual Environment
    setup_virtual_environment()

    # Step 2: Check & Install dependencies
    if not check_dependencies():
        install_dependencies()

    # Step 3: Run the web application
    venv_python = get_venv_python()
    app_path = os.path.join("website", "app.py")
    
    if not os.path.exists(app_path):
        print_status(f"Application entry point not found at '{app_path}'!", "error")
        input("Press Enter to exit...")
        sys.exit(1)

    print_status("Starting Flask web server...", "info")
    # Run server process in the background, redirecting output
    server_process = subprocess.Popen(
        [venv_python, app_path],
        cwd=os.getcwd()
    )

    # Step 4: Wait for server to start, then open browser
    if wait_for_server(port=5000):
        print_status("Server is up and running on http://localhost:5000", "success")
        
        chrome_path = find_chrome()
        if chrome_path:
            print_status(f"Opening Chrome in web app mode...", "success")
            subprocess.Popen([chrome_path, "--app=http://localhost:5000"])
        else:
            print_status("Chrome not found. Opening default web browser...", "warning")
            webbrowser.open("http://localhost:5000")
    else:
        print_status("Server failed to start or respond in time. Please check logs.", "error")

    # Let the launcher process block on the server process so that:
    # 1. The terminal window stays open, showing Flask stdout/stderr.
    # 2. Closing the terminal window kills the Flask process.
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print_status("\nShutting down server...", "info")
        server_process.terminate()

if __name__ == "__main__":
    main()
