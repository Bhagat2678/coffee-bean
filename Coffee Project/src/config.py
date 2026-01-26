"""
Module: config.py
Description: Configuration settings for the coffee bean analysis project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "output"

# Create directories if they don't exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Database Configuration
# SQLite is used by default (file-based, no server needed)
# Change DATABASE_URL to use PostgreSQL or MySQL if needed
DATABASE_URL = os.getenv(
    'DATABASE_URL', 
    f'sqlite:///{PROJECT_ROOT / "coffee_analysis.db"}'
)

# Alternative database URLs (uncomment to use):
# PostgreSQL:
# DATABASE_URL = 'postgresql://user:password@localhost/coffee_db'
# MySQL:
# DATABASE_URL = 'mysql+pymysql://user:password@localhost/coffee_db'

# Image Processing Settings
MIN_BEAN_AREA = int(os.getenv('MIN_BEAN_AREA', 100))
BLUR_KERNEL_SIZE = int(os.getenv('BLUR_KERNEL_SIZE', 7))
MORPHOLOGY_ITERATIONS = int(os.getenv('MORPHOLOGY_ITERATIONS', 2))

# Logging Settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = PROJECT_ROOT / "logs" / "coffee_analysis.log"

# Create logs directory if it doesn't exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Model Settings (if using YOLO)
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.45))
