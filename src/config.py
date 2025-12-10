# src/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_like):
    p = Path(path_like)
    return p if p.is_absolute() else PROJECT_ROOT / p


# Data layout
DATA_DIR = _resolve(os.getenv("DATA_DIR", "data"))
RAW_DIR = _resolve(os.getenv("RAW_DIR", DATA_DIR / "raw"))
SENTINEL_DIR = _resolve(os.getenv("SENTINEL_DIR", RAW_DIR / "sentinel"))
CDL_DIR = _resolve(os.getenv("CDL_DIR", RAW_DIR / "cdl"))
INTERIM_DIR = _resolve(os.getenv("INTERIM_DIR", DATA_DIR / "interim"))
FEATURES_DIR = _resolve(os.getenv("FEATURES_DIR", DATA_DIR / "features"))

# Copernicus Data Space credentials
CDSE_CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CDSE_CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")

def _to_bool(val: str) -> bool:
    return str(val).lower() in {"1", "true", "yes", "on"}

# Hardware / execution toggles
GPU_ENABLED = _to_bool(os.getenv("GPU_ENABLED", "false"))
TARGET_CROP = os.getenv("TARGET_CROP", "corn").lower()
S2_SOURCE = os.getenv("S2_SOURCE", "cdse").lower()

# CDL crop codes
CROP_NAME_TO_CODE = {
    "corn": 1,
    "soybean": 5,
    "beans": 5,
    "alfalfa": 36,
    "wheat": 23,  # winter wheat
}


def get_crop_code(name: str) -> int:
    key = str(name).lower()
    if key not in CROP_NAME_TO_CODE:
        raise ValueError(f"Unknown crop '{name}'. Supported: {list(CROP_NAME_TO_CODE.keys())}")
    return CROP_NAME_TO_CODE[key]


def get_target_crop_code() -> int:
    return get_crop_code(TARGET_CROP)


def get_season_windows_for_crop(crop: str):
    crop = crop.lower()
    # 5 high-impact phenology windows per crop (Nebraska-oriented)
    windows = {
        "corn": [
            (5, 20, 5, 31),  # V4–V6
            (6, 10, 6, 25),  # rapid vegetative
            (7, 1, 7, 10),   # pre-tassel
            (7, 20, 7, 31),  # tassel/silking
            (8, 5, 8, 20),   # early grain fill
        ],
        "soybean": [
            (5, 20, 5, 31),  # V2–V3
            (6, 10, 6, 25),  # vegetative expansion
            (7, 1, 7, 10),   # R1 start
            (7, 20, 7, 31),  # R3 pod set
            (8, 5, 8, 20),   # R5 seed fill
        ],
        "beans": [
            (5, 20, 5, 31),
            (6, 10, 6, 25),
            (7, 1, 7, 10),
            (7, 20, 7, 31),
            (8, 5, 8, 20),
        ],
        "alfalfa": [
            (5, 15, 5, 25),  # after first cut regrowth
            (6, 10, 6, 20),  # second cut window
            (7, 5, 7, 15),   # third cut window
            (8, 1, 8, 15),   # late summer regrowth
            (9, 5, 9, 15),   # fall regrowth
        ],
        "wheat": [
            (3, 15, 3, 31),  # greenup
            (4, 10, 4, 25),  # stem elongation
            (5, 10, 5, 25),  # boot
            (6, 5, 6, 20),   # heading/grain fill
            (7, 1, 7, 10),   # pre-harvest
        ],
    }
    if crop not in windows:
        raise ValueError(f"No season windows defined for crop '{crop}'")
    return windows[crop]

# Copernicus Data Space endpoints (Sentinel Hub over CDSE)
CDSE_BASE_URL = os.getenv("CDSE_BASE_URL", "https://sh.dataspace.copernicus.eu")
CDSE_AUTH_BASE_URL = os.getenv(
    "CDSE_AUTH_BASE_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect",
)
CDSE_TOKEN_URL = os.getenv(
    "CDSE_TOKEN_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
)
