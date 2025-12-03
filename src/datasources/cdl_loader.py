# src/datasources/cdl_loader.py
import rasterio
import numpy as np
from pathlib import Path
from src.config import CDL_DIR

CORN_CLASS = 1  # example; verify from CDL legend PDF

def load_cdl_year(year: int):
    path = CDL_DIR / f"cdl_NE_{year}.tif"
    with rasterio.open(path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
    return data, transform, crs

def build_stable_corn_mask_from_years(years):
    cdls = []

    for y in years:
        data, transform, crs = load_cdl_year(y)
        cdls.append(data)

    cdls = np.stack(cdls, axis=0)  # shape: [T, H, W]

    # Stability is computed ONLY across the years provided
    same_crop = np.all(cdls == cdls[0], axis=0)

    # Corn-only filter applied to FIRST TRAINING YEAR ONLY
    stable_corn_mask = same_crop & (cdls[0] == CORN_CLASS)

    return stable_corn_mask.astype(np.uint8), transform, crs
