# src/datasources/cdl_loader.py

import argparse
import gzip
import shutil
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

from src.config import CDL_DIR


CORN_CLASS = 1


def get_cdl_url(year: int) -> str:
    # USDA Cropland Data Layer public release pattern (GeoTIFF gzipped)
    return f"https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/{year}_30m_cdls.tif.gz"


def download_cdl_year(year: int, overwrite: bool = False) -> Path:
    CDL_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = CDL_DIR / f"cdl_NE_{year}.tif.gz"
    tif_path = CDL_DIR / f"cdl_NE_{year}.tif"

    if tif_path.exists() and not overwrite:
        return tif_path

    url = get_cdl_url(year)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    with open(gz_path, "wb") as f:
        shutil.copyfileobj(resp.raw, f)

    with gzip.open(gz_path, "rb") as f_in, open(tif_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    gz_path.unlink(missing_ok=True)
    return tif_path


def load_cdl_year(year: int, aoi=None):
    path = CDL_DIR / f"cdl_NE_{year}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Missing CDL for {year}: {path}")

    with rasterio.open(path) as src:
        if aoi is not None:
            bounds_proj = transform_bounds(
                CRS.from_epsg(4326), src.crs, *aoi, densify_pts=21
            )
            window = from_bounds(*bounds_proj, transform=src.transform)
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


def main():
    parser = argparse.ArgumentParser(description="Download CDL GeoTIFFs for Nebraska subset")
    parser.add_argument("--years", nargs="+", type=int, required=True, help="Years to download (e.g., 2019 2020)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for y in args.years:
        path = download_cdl_year(y, overwrite=args.overwrite)
        print(f"CDL saved: {path}")


if __name__ == "__main__":
    main()
