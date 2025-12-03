from src.config import FEATURE_DIR, INTERIM_DIR, SENTINEL_DIR, CDL_DIR

def ensure_directories():
    for d in [FEATURE_DIR, INTERIM_DIR, SENTINEL_DIR, CDL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
