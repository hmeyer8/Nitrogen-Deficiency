# src/datasources/sentinel_download.py
from datetime import date
from pathlib import Path

import numpy as np
from sentinelhub import (
    BBox, CRS, DataCollection, SentinelHubRequest,
    MimeType, bbox_to_dimensions
)

from src.datasources.copernicus_client import get_sh_config
from src.config import SENTINEL_DIR

# S2 spectral + cloud mask
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

def download_sentinel_cube(aoi_gdf, time_interval, out_path: Path):
    """Download a small cube for given AOI (GeoDataFrame) and time interval."""
    config = get_sh_config()

    aoi = aoi_gdf.to_crs(epsg=4326).unary_union
    minx, miny, maxx, maxy = aoi.bounds
    bbox = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)

    resolution = 10  # meters
    size_x, size_y = bbox_to_dimensions(bbox, resolution=resolution)

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

    data = request.get_data()[0]  # shape: H x W x 11
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, data)
    return out_path
