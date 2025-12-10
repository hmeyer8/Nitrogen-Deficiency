# src/datasources/sentinel_download.py

import argparse
import math
import os
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from sentinelhub import BBox, CRS, MimeType, SentinelHubRequest, bbox_to_dimensions

from src.config import SENTINEL_DIR, get_target_crop_code, get_season_windows_for_crop, TARGET_CROP
from src.datasources.copernicus_client import CDSE_S2_L2A, get_sh_config
from src.datasources.cdl_loader import build_stable_crop_mask_from_years
from src.geo.aoi_nebraska import NEBRASKA_BBOX


EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"]
    }],
    output: {
      bands: 11,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  return [
    sample.B02, sample.B03, sample.B04,
    sample.B05, sample.B06, sample.B07,
    sample.B08, sample.B8A, sample.B11, sample.B12,
    sample.SCL
  ];
}
"""


RESOLUTION_METERS = 60  # downsample to keep request sizes reasonable
MAX_TILE_SIZE = 2000    # Sentinel Hub per-request limit is 2500 px; stay under it
S2_SOURCE = os.getenv("S2_SOURCE", "cdse").lower()  # cdse | pc
STABLE_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


def download_sentinel_cube(year: int, verbose: bool = False):
    if S2_SOURCE == "pc":
        # Delegate to PC downloader (uses stable crop mask AOI and crop-specific windows)
        from src.datasources.sentinel_download_pc import download_pc
        return download_pc(year=year, verbose=verbose)
    return download_sentinel_cube_cdse(year=year, verbose=verbose)


def download_sentinel_cube_cdse(year: int, verbose: bool = False):
    config = get_sh_config()

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
    transformer = Transformer.from_crs(mask_crs, CRS.WGS84.pyproj_crs(), always_xy=True)
    minx, miny = transformer.transform(xmin_proj, ymin_proj)
    maxx, maxy = transformer.transform(xmax_proj, ymax_proj)

    bbox_wgs84 = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)
    # Work in a projected CRS so resolution in meters matches the request
    bbox = bbox_wgs84.transform(CRS.POP_WEB)

    full_width, full_height = bbox_to_dimensions(bbox, resolution=RESOLUTION_METERS)
    tiles_x = math.ceil(full_width / MAX_TILE_SIZE)
    tiles_y = math.ceil(full_height / MAX_TILE_SIZE)

    crop_windows = get_season_windows_for_crop(TARGET_CROP)
    time_windows = [
        (date(year, m1, d1), date(year, m2, d2)) for (m1, d1, m2, d2) in crop_windows
    ]

    if verbose:
        print("Requesting Sentinel-2 L2A time series")
        print("Year:", year)
        print("Windows:", time_windows)
        print("AOI:", bbox_wgs84)
        print("Resolution (m):", RESOLUTION_METERS)
        print("Full raster size:", full_width, full_height)
        print("Grid tiles (y, x):", tiles_y, tiles_x)

    cube = np.empty((len(time_windows), full_height, full_width, 11), dtype=np.float32)

    pixel_size_x = (bbox.max_x - bbox.min_x) / full_width
    pixel_size_y = (bbox.max_y - bbox.min_y) / full_height

    col_widths = []
    remaining_w = full_width
    while remaining_w > 0:
        w = min(MAX_TILE_SIZE, remaining_w)
        col_widths.append(w)
        remaining_w -= w

    row_heights = []
    remaining_h = full_height
    while remaining_h > 0:
        h = min(MAX_TILE_SIZE, remaining_h)
        row_heights.append(h)
        remaining_h -= h

    total_tiles = len(col_widths) * len(row_heights)
    if verbose:
        print("Column widths:", col_widths)
        print("Row heights:", row_heights)
        print("Total tiles:", total_tiles)

    for t, time_interval in enumerate(time_windows):
        if verbose:
            print(f"\nWindow {t+1}/{len(time_windows)}: {time_interval}")
        tile_idx = 0
        y_offset = 0
        for row_h in row_heights:
            x_offset = 0
            y_top = bbox.max_y - y_offset * pixel_size_y
            y_bottom = y_top - row_h * pixel_size_y

            for col_w in col_widths:
                x_left = bbox.min_x + x_offset * pixel_size_x
                x_right = x_left + col_w * pixel_size_x

                tile_bbox = BBox((x_left, y_bottom, x_right, y_top), crs=CRS.POP_WEB)
                size = (col_w, row_h)

                request = SentinelHubRequest(
                    evalscript=EVALSCRIPT,
                    input_data=[SentinelHubRequest.input_data(
                        data_collection=CDSE_S2_L2A,
                        time_interval=time_interval,
                        mosaicking_order="mostRecent"
                    )],
                    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                    bbox=tile_bbox,
                    size=size,
                    config=config,
                )

                tile = request.get_data()[0]
                th, tw = tile.shape[:2]

                y0 = y_offset
                y1 = y_offset + th
                x0 = x_offset
                x1 = x_offset + tw

                cube[t, y0:y1, x0:x1] = tile

                tile_idx += 1
                if verbose:
                    print(f"Tile {tile_idx}/{total_tiles}: bbox={tile_bbox}, size=({tw},{th}) placed at ({y0}:{y1}, {x0}:{x1})")

                x_offset += col_w
            y_offset += row_h

    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SENTINEL_DIR / f"s2_ne_{year}.npy"
    np.save(out_path, cube)

    if verbose:
        print("Saved:", out_path)
        print("Array shape:", cube.shape)

    return out_path


def download_sentinel_cube_pc(year: int, verbose: bool = False):
    """
    Download Sentinel-2 L2A from Planetary Computer / AWS sentinel-cogs.
    Produces a time stack matching the CDSE shape: (T, H, W, 11) at 60m.
    """
    from shapely.geometry import box
    from pystac_client import Client
    import stackstac
    import dask.array as da

    crop_windows = get_season_windows_for_crop(TARGET_CROP)
    time_windows = [
        (date(year, m1, d1), date(year, m2, d2)) for (m1, d1, m2, d2) in crop_windows
    ]

    aoi_geom = box(*NEBRASKA_BBOX).__geo_interface__
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    arrays = []
    for idx, (start, end) in enumerate(time_windows):
        search = client.search(
            collections=["sentinel-2-l2a"],
            intersects=aoi_geom,
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": 30}},
            max_items=3,
        )
        items = list(search.get_items())
        if not items:
            raise RuntimeError(f"No Sentinel-2 items for {start} to {end}")
        cube = stackstac.stack(
            items,
            assets=["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"],
            resolution=RESOLUTION_METERS,
            chunksize=1024,
        )
        # Simple mosaic: mean across available scenes in the window
        cube = cube.mean("time")
        arrays.append(cube.data)
        if verbose:
            print(f"Window {idx+1}/{len(time_windows)} stacked from {len(items)} scenes.")

    stack = np.stack([arr.compute() for arr in arrays], axis=0).astype(np.float32)
    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SENTINEL_DIR / f"s2_ne_{year}.npy"
    np.save(out_path, stack)
    if verbose:
        print("Saved:", out_path, "shape:", stack.shape)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    download_sentinel_cube(year=args.year, verbose=args.verbose)


if __name__ == "__main__":
    main()
