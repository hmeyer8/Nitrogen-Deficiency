# src/features/indices.py
import numpy as np


def apply_scl_mask(cube: np.ndarray) -> np.ndarray:
    """
    Apply Sentinel-2 SCL mask and keep band alignment.
    Returns a cube with spectral bands masked (np.nan) and SCL unchanged in the last band.
    """
    scl = cube[..., -1]
    spectral = cube[..., :-1].copy()
    invalid = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 11)
    spectral[invalid] = np.nan
    return np.concatenate([spectral, scl[..., None]], axis=-1)


def compute_ndvi(s2_cube: np.ndarray) -> np.ndarray:
    # cube: H x W x 10 (spectral bands only)
    red = s2_cube[..., 2]   # B04
    nir = s2_cube[..., 6]   # B08
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi


def compute_ndre(s2_cube: np.ndarray) -> np.ndarray:
    nir = s2_cube[..., 7]       # B8A narrow NIR
    red_edge = s2_cube[..., 3]  # B05
    ndre = (nir - red_edge) / (nir + red_edge + 1e-6)
    return ndre
