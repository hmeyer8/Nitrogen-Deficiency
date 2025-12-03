# src/datasources/copernicus_client.py
from sentinelhub import SHConfig

from src.config import CDSE_CLIENT_ID, CDSE_CLIENT_SECRET

def get_sh_config() -> SHConfig:
    config = SHConfig()
    if CDSE_CLIENT_ID and CDSE_CLIENT_SECRET:
        config.sh_client_id = CDSE_CLIENT_ID
        config.sh_client_secret = CDSE_CLIENT_SECRET
    else:
        raise RuntimeError("Copernicus credentials not set in .env")
    return config