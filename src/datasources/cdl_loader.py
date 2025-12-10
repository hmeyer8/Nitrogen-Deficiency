# src/datasources/cdl_loader.py

import argparse
import gzip
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

from src.config import CDL_DIR


CORN_CLASS = 1


def get_cdl_url_candidates(year: int):
    """
    USDA Cropland Data Layer release filenames move around occasionally.
    Try a handful of known patterns before failing.
    """
    base = "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets"
    return [
        f"{base}/{year}_30m_cdls.zip",         # common current pattern (e.g., 2020)
        f"{base}/CDL_{year}_30m_cdls.zip",     # some years add CDL_ prefix
        f"{base}/{year}_30m_cdls.tif.gz",      # gz variant without prefix
        f"{base}/CDL_{year}_30m_cdls.tif.gz",  # gz variant with prefix
    ]


def download_cdl_year(year: int, overwrite: bool = False) -> Path:
    CDL_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = CDL_DIR / f"cdl_NE_{year}.tif.gz"
    tif_path = CDL_DIR / f"cdl_NE_{year}.tif"
    zip_path = CDL_DIR / f"cdl_NE_{year}.zip"

    if tif_path.exists() and not overwrite:
        return tif_path

    last_error = None
    for url in get_cdl_url_candidates(year):
        suffix = ".zip" if url.endswith(".zip") else ".gz"
        target_path = zip_path if suffix == ".zip" else gz_path
        existing_tifs = set(p.name for p in CDL_DIR.glob("*.tif"))

        for attempt in range(1, 4):
            tmp_path = None
            try:
                resp = requests.get(url, stream=True, timeout=(15, 300))
                resp.raise_for_status()

                tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=CDL_DIR)
                with os.fdopen(tmp_fd, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

                Path(tmp_path).replace(target_path)
                tmp_path = target_path
            except Exception as e:
                last_error = e
                if tmp_path and Path(tmp_path).exists():
                    Path(tmp_path).unlink(missing_ok=True)
                if attempt < 3:
                    continue  # retry same URL
                else:
                    break  # move to next URL

            print(f"Downloaded {year} from {url} (attempt {attempt})")

            if suffix == ".zip":
                shutil.unpack_archive(target_path, CDL_DIR)
                target_path.unlink(missing_ok=True)
                new_tifs = [p for p in CDL_DIR.glob("*.tif") if p.name not in existing_tifs]
                if not new_tifs:
                    raise RuntimeError(f"ZIP extracted but no new .tif found for {year}")
                if len(new_tifs) > 1:
                    candidates = [p for p in new_tifs if str(year) in p.name]
                    chosen = candidates[0] if candidates else new_tifs[0]
                else:
                    chosen = new_tifs[0]
                chosen.replace(tif_path)
            else:
                with gzip.open(target_path, "rb") as f_in, open(tif_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                target_path.unlink(missing_ok=True)

            break  # exit attempt loop on success
        else:
            # Exhausted retries for this URL; try next candidate
            continue

        # Success for one URL; exit outer loop
        break
    else:
        # No URL succeeded
        raise RuntimeError(f"Failed to download CDL for {year}; last error: {last_error}")

    if not tif_path.exists():
        raise RuntimeError(f"Download pipeline completed but {tif_path} does not exist")

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


def build_stable_crop_mask_from_years(years, crop_class: int, aoi=None):
    cdls = []

    for y in years:
        data, transform, crs = load_cdl_year(y, aoi=aoi)
        cdls.append(data)

    cdls = np.stack(cdls, axis=0)
    same_crop = np.all(cdls == cdls[0], axis=0)
    stable_mask = same_crop & (cdls[0] == crop_class)
    return stable_mask.astype(np.uint8), transform, crs


def build_stable_corn_mask_from_years(years, aoi=None):
    return build_stable_crop_mask_from_years(years=years, crop_class=CORN_CLASS, aoi=aoi)


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
