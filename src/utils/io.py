from src.config import DATA_DIR, RAW_DIR, INTERIM_DIR, FEATURES_DIR


def ensure_directories():
    """Ensure all required data directories exist."""
    for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, FEATURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
