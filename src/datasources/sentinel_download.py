# src/datasources/sentinel_download.py

import argparse
from datetime import date
from pathlib import Path
import numpy as np

from sentinelhub import (
    BBox, CRS, DataCollection, SentinelHubRequest,
    MimeType, bbox_to_dimensions
)

from src.datasources.copernicus_client import get_sh_config
from src.geo.aoi_nebraska import NEBRASKA_BBOX
from src.config import SENTINEL_DIR


EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"],
      units: "REFLECTANCE"
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


def download_sentinel_cube(year: int, verbose: bool = False):

    config = get_sh_config()

    minx, miny, maxx, maxy = NEBRASKA_BBOX
    bbox = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)

    resolution = 10
    size_x, size_y = bbox_to_dimensions(bbox, resolution=resolution)

    time_interval = (
        date(year, 6, 1),
        date(year, 8, 31)
    )

    if verbose:
        print("Requesting Sentinel-2 L2A")
        print("Year:", year)
        print("AOI:", bbox)
        print("Pixel size:", size_x, size_y)

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
            mosaicking_order="mostRecent"
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(size_x, size_y),
        config=config,
    )

    data = request.get_data()[0]

    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SENTINEL_DIR / f"s2_ne_{year}.npy"
    np.save(out_path, data)

    if verbose:
        print("Saved:", out_path)
        print("Array shape:", data.shape)

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    download_sentinel_cube(year=args.year, verbose=args.verbose)


if __name__ == "__main__":
    main()
