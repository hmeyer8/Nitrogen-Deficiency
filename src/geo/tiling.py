# src/geo/tiling.py
import numpy as np

def generate_tile_coords(mask: np.ndarray, tile_size=32, stride=32, max_tiles=None):
    H, W = mask.shape
    coords = []
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            tile = mask[y:y+tile_size, x:x+tile_size]
            if tile.mean() > 0.95:  # >95% of pixels stable corn
                coords.append((y, x))
    if max_tiles:
        coords = coords[:max_tiles]
    return coords
