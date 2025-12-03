# src/geo/aoi_nebraska.py
import geopandas as gpd
from src.datasources.cdl_loader import build_stable_corn_mask, mask_to_polygons

def get_nebraska_stable_corn_aoi():
    mask, transform, crs = build_stable_corn_mask()
    gdf = mask_to_polygons(mask, transform, crs)
    return gdf
