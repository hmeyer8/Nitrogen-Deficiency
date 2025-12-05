# src/datasources/copernicus_client.py
from sentinelhub import DataCollection, SHConfig

from src.config import (
    CDSE_AUTH_BASE_URL,
    CDSE_BASE_URL,
    CDSE_CLIENT_ID,
    CDSE_CLIENT_SECRET,
    CDSE_TOKEN_URL,
)


CDSE_S2_L2A = DataCollection.SENTINEL2_L2A.define_from(
    name="SENTINEL2_L2A_CDSE",
    service_url=CDSE_BASE_URL,
)


def get_sh_config() -> SHConfig:
    """
    Configure Sentinel Hub to talk to Copernicus Data Space (CDSE).
    """
    if not (CDSE_CLIENT_ID and CDSE_CLIENT_SECRET):
        raise RuntimeError("Copernicus credentials not set in .env")

    config = SHConfig()
    config.sh_client_id = CDSE_CLIENT_ID
    config.sh_client_secret = CDSE_CLIENT_SECRET
    config.sh_base_url = CDSE_BASE_URL
    config.sh_auth_base_url = CDSE_AUTH_BASE_URL
    config.sh_token_url = CDSE_TOKEN_URL
    return config
