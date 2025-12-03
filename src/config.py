# src/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
SENTINEL_DIR = DATA_DIR / "raw" / "sentinel"
CDL_DIR = DATA_DIR / "raw" / "cdl"
INTERIM_DIR = DATA_DIR / "interim"
FEATURE_DIR = DATA_DIR / "features"

CDSE_CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CDSE_CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")
