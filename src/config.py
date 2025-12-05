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
