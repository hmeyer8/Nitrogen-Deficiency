# src/datasources/cdl_loader.py

import rasterio
import numpy as np
from pathlib import Path
from rasterio.windows import from_bounds

from src.config import CDL_DIR
from src.geo.aoi_nebraska import NEBRASKA_BBOX


CORN_CLASS = 1


def load_cdl_year(year: int, aoi=None):
    path = CDL_DIR / f"cdl_NE_{year}.tif"

    with rasterio.open(path) as src:

        if aoi is not None:
            window = from_bounds(*aoi, transform=src.transform)
            data = src.read(1, window=window)
            transform = src.window_transform(window)
        else:
            data = src.read(1)
            transform = src.transform

        crs = src.crs

    return data, transform, crs


def build_stable_corn_mask_from_years(years, aoi=None):
    cdls = []

    for y in years:
        data, transform, crs = load_cdl_year(y, aoi=aoi)
        cdls.append(data)

    cdls = np.stack(cdls, axis=0)

    same_crop = np.all(cdls == cdls[0], axis=0)
    stable_corn_mask = same_crop & (cdls[0] == CORN_CLASS)

    return stable_corn_mask.astype(np.uint8), transform, crs
