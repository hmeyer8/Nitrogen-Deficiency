# src/features/indices.py
import numpy as np

def apply_scl_mask(cube):
    scl = cube[..., -1]
    valid = ~((scl == 3) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 11))
    return cube[..., :-1] * valid[..., None]

def compute_ndvi(s2_cube: np.ndarray) -> np.ndarray:
    # cube: H x W x 11 (band order as in evalscript)
    red = s2_cube[..., 2]   # B04
    nir = s2_cube[..., 6]   # B08
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

def compute_ndre(s2_cube: np.ndarray) -> np.ndarray:
    nir = s2_cube[..., 7]   # B8A narrow NIR
    red_edge = s2_cube[..., 4]  # B05
    ndre = (nir - red_edge) / (nir + red_edge + 1e-6)
    return ndre
