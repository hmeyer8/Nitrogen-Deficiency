# Download Sentinel-2 L2A time stacks from the Planetary Computer / AWS sentinel-cogs.
# Outputs: data/raw/sentinel/s2_ne_{year}.npy with shape (T, H, W, 11).
import argparse
import gc
import itertools
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import dask.array as da
import rasterio
from shapely.geometry import box
from pyproj import Transformer
from pystac_client import Client
import stackstac
import planetary_computer

from src.config import SENTINEL_DIR
from src.geo.aoi_nebraska import NEBRASKA_BBOX
from src.config import get_target_crop_code, get_season_windows_for_crop, TARGET_CROP
from src.datasources.cdl_loader import build_stable_crop_mask_from_years

# Keep memory low: modest chunking and single-worker writes.
RESOLUTION_METERS = 60
STORE_WORKERS = int(os.getenv("PC_STORE_WORKERS", "1"))
CHUNK_SIZE = int(os.getenv("PC_CHUNK_SIZE", "256"))
STABLE_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
ASSETS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"]


def _search_items(client, aoi_geom, start, end, max_cloud, max_items):
    search = client.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi_geom,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return list(itertools.islice(search.items(), max_items))


def download_pc(
    year: int,
    verbose: bool = False,
    max_cloud: int = 30,
    max_items: int = 1,
    relax_days: int = 7,
    relax_cloud: int = 80,
):
    crop_code = get_target_crop_code()
    stable_mask, mask_transform, mask_crs = build_stable_crop_mask_from_years(
        years=STABLE_YEARS, crop_class=crop_code, aoi=NEBRASKA_BBOX
    )
    rows, cols = np.nonzero(stable_mask)
    if rows.size == 0:
        raise RuntimeError("Stable crop mask is empty; cannot define AOI for download.")
    row_min, row_max = rows.min(), rows.max() + 1
    col_min, col_max = cols.min(), cols.max() + 1
    xs = [col_min, col_max]
    ys = [row_min, row_max]
    xs_geo, ys_geo = rasterio.transform.xy(mask_transform, ys, xs, offset="ul")
    xmin_proj, xmax_proj = min(xs_geo), max(xs_geo)
    ymin_proj, ymax_proj = min(ys_geo), max(ys_geo)
    to_3857 = Transformer.from_crs(mask_crs, "EPSG:3857", always_xy=True)
    to_wgs = Transformer.from_crs(mask_crs, "EPSG:4326", always_xy=True)
    minx_3857, miny_3857 = to_3857.transform(xmin_proj, ymin_proj)
    maxx_3857, maxy_3857 = to_3857.transform(xmax_proj, ymax_proj)
    minx_wgs, miny_wgs = to_wgs.transform(xmin_proj, ymin_proj)
    maxx_wgs, maxy_wgs = to_wgs.transform(xmax_proj, ymax_proj)

    aoi_geom = box(minx_wgs, miny_wgs, maxx_wgs, maxy_wgs).__geo_interface__
    bounds_3857 = (minx_3857, miny_3857, maxx_3857, maxy_3857)
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    crop_windows = get_season_windows_for_crop(TARGET_CROP)
    time_windows = [
        (date(year, m1, d1), date(year, m2, d2)) for (m1, d1, m2, d2) in crop_windows
    ]

    # First pass: compute each window, track max dims (allow missing windows)
    window_arrays = []
    max_h = max_w = None
    for idx, (start, end) in enumerate(time_windows):
        items = _search_items(client, aoi_geom, start, end, max_cloud, max_items)
        if not items:
            # Relax window and cloud threshold
            relaxed_start = max(date(year, 1, 1), start - timedelta(days=relax_days))
            relaxed_end = min(date(year, 12, 31), end + timedelta(days=relax_days))
            items = _search_items(client, aoi_geom, relaxed_start, relaxed_end, relax_cloud, max_items)
        if not items:
            if verbose:
                print(f"Window {idx+1} has no scenes after relaxing; filling with NaNs.")
            window_arrays.append(None)
            continue

        if verbose:
            print(f"Window {idx+1}/{len(time_windows)}: {start} to {end}, items={len(items)}")

        signed_items = [planetary_computer.sign(item) for item in items]
        cube = stackstac.stack(
            signed_items,
            assets=ASSETS,
            resolution=RESOLUTION_METERS,
            epsg=3857,
            bounds=bounds_3857,
            chunksize=CHUNK_SIZE,
        )
        cube = cube.mean("time").transpose("y", "x", "band")
        cube = cube.chunk({"y": CHUNK_SIZE, "x": CHUNK_SIZE})
        arr_da = cube.data.astype(np.float32)  # dask array (y, x, band)
        window_arrays.append(arr_da)
        h, w, _ = arr_da.shape
        max_h = h if max_h is None else max(max_h, h)
        max_w = w if max_w is None else max(max_w, w)

    if max_h is None or max_w is None:
        raise RuntimeError(
            "No Sentinel-2 scenes found for any window (even after relaxation). "
            "Try increasing --max-items, --relax-days/--relax-cloud, or broadening the AOI."
        )

    # Allocate memmap and pad as needed
    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SENTINEL_DIR / f"s2_ne_{year}.npy"
    memmap = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(time_windows), max_h, max_w, len(ASSETS)),
    )
    memmap[...] = np.nan

    for idx, arr in enumerate(window_arrays):
        if arr is None:
            continue
        h, w, b = arr.shape
        target = memmap[idx, :h, :w, :b]
        try:
            da.store(arr, target, lock=False, compute=True, num_workers=STORE_WORKERS)
        except Exception as e:
            if verbose:
                print(f"Store failed for window {idx+1}: {e}. Filling with NaNs and continuing.")
            memmap[idx, ...] = np.nan
        del arr, target
        gc.collect()

    if verbose:
        print("Saved:", out_path, "shape:", memmap.shape)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-cloud", type=int, default=30, help="Max cloud cover percent per scene")
    parser.add_argument("--max-items", type=int, default=1, help="Max scenes per window to mosaic (keep at 1 to avoid large stacks)")
    parser.add_argument("--relax-days", type=int, default=7, help="Fallback: expand window by this many days if empty")
    parser.add_argument("--relax-cloud", type=int, default=80, help="Fallback: cloud threshold when relaxed search kicks in")
    args = parser.parse_args()
    download_pc(
        year=args.year,
        verbose=args.verbose,
        max_cloud=args.max_cloud,
        max_items=args.max_items,
        relax_days=args.relax_days,
        relax_cloud=args.relax_cloud,
    )


if __name__ == "__main__":
    main()
