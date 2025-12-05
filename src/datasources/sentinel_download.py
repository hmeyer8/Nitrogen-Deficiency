# src/datasources/sentinel_download.py

import argparse
import math
from datetime import date
from pathlib import Path

import numpy as np
from sentinelhub import BBox, CRS, MimeType, SentinelHubRequest, bbox_to_dimensions

from src.config import SENTINEL_DIR
from src.datasources.copernicus_client import CDSE_S2_L2A, get_sh_config
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


def download_sentinel_cube(year: int, verbose: bool = False):
    config = get_sh_config()

    minx, miny, maxx, maxy = NEBRASKA_BBOX
    bbox_wgs84 = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)
    # Work in a projected CRS so resolution in meters matches the request
    bbox = bbox_wgs84.transform(CRS.POP_WEB)

    full_width, full_height = bbox_to_dimensions(bbox, resolution=RESOLUTION_METERS)
    tiles_x = math.ceil(full_width / MAX_TILE_SIZE)
    tiles_y = math.ceil(full_height / MAX_TILE_SIZE)

    time_interval = (
        date(year, 6, 1),
        date(year, 8, 31)
    )

    if verbose:
        print("Requesting Sentinel-2 L2A")
        print("Year:", year)
        print("AOI:", bbox_wgs84)
        print("Resolution (m):", RESOLUTION_METERS)
        print("Full raster size:", full_width, full_height)
        print("Grid tiles (y, x):", tiles_y, tiles_x)

    cube = np.empty((full_height, full_width, 11), dtype=np.float32)

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

            cube[y0:y1, x0:x1] = tile

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    download_sentinel_cube(year=args.year, verbose=args.verbose)


if __name__ == "__main__":
    main()
