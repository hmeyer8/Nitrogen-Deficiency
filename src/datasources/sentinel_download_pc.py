# Download Sentinel-2 L2A time stacks from the Planetary Computer / AWS sentinel-cogs.
# Outputs: data/raw/sentinel/s2_ne_{year}.npy with shape (T, H, W, 11).
import argparse
import gc
import itertools
from datetime import date
from pathlib import Path

import numpy as np
from shapely.geometry import box
from pyproj import Transformer
from pystac_client import Client
import stackstac
import planetary_computer

from src.config import SENTINEL_DIR
from src.geo.aoi_nebraska import NEBRASKA_BBOX

RESOLUTION_METERS = 60
SEASON_WINDOWS = [
    (6, 1, 6, 30),
    (7, 1, 7, 31),
    (8, 1, 8, 31),
]
ASSETS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"]


def download_pc(year: int, verbose: bool = False, max_cloud: int = 30, max_items: int = 3):
    aoi_geom = box(*NEBRASKA_BBOX).__geo_interface__
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    minx, miny, maxx, maxy = NEBRASKA_BBOX
    bx0, by0 = transformer.transform(minx, miny)
    bx1, by1 = transformer.transform(maxx, maxy)
    bounds_3857 = (min(bx0, bx1), min(by0, by1), max(bx0, bx1), max(by0, by1))
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    time_windows = [
        (date(year, m1, d1), date(year, m2, d2)) for (m1, d1, m2, d2) in SEASON_WINDOWS
    ]

    # First pass: compute each window, track max dims
    window_arrays = []
    max_h = max_w = None
    for idx, (start, end) in enumerate(time_windows):
        search = client.search(
            collections=["sentinel-2-l2a"],
            intersects=aoi_geom,
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
        )
        items = list(itertools.islice(search.items(), max_items))
        if not items:
            raise RuntimeError(f"No Sentinel-2 items for {start} to {end}")

        if verbose:
            print(f"Window {idx+1}/{len(time_windows)}: {start} to {end}, items={len(items)}")

        signed_items = [planetary_computer.sign(item) for item in items]
        cube = stackstac.stack(
            signed_items,
            assets=ASSETS,
            resolution=RESOLUTION_METERS,
            epsg=3857,
            bounds=bounds_3857,
        )
        cube = cube.mean("time")
        arr = cube.data.astype(np.float32).compute()
        window_arrays.append(arr)
        h, w, _ = arr.shape
        max_h = h if max_h is None else max(max_h, h)
        max_w = w if max_w is None else max(max_w, w)

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
        h, w, b = arr.shape
        memmap[idx, :h, :w, :b] = arr
        del arr
        gc.collect()

    if verbose:
        print("Saved:", out_path, "shape:", memmap.shape)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-cloud", type=int, default=30, help="Max cloud cover percent per scene")
    parser.add_argument("--max-items", type=int, default=3, help="Max scenes per window to mosaic")
    args = parser.parse_args()
    download_pc(year=args.year, verbose=args.verbose, max_cloud=args.max_cloud, max_items=args.max_items)


if __name__ == "__main__":
    main()
