# src/experiments/prepare_dataset.py

import numpy as np
import hashlib
from pathlib import Path

from src.config import INTERIM_DIR, SENTINEL_DIR
from src.datasources.cdl_loader import build_stable_corn_mask_from_years
from src.geo.tiling import generate_tile_coords, extract_tiles_from_cube
from src.features.indices import compute_ndvi, compute_ndre, apply_scl_mask
from src.utils.io import ensure_directories
from src.geo.aoi_nebraska import NEBRASKA_BBOX


TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]
TEST_YEAR = 2024

TILE_SIZE = 32
STRIDE = 32
MAX_TILES = 2000


def hash_coords(coords):
    h = hashlib.sha256()
    h.update(str(coords).encode())
    return h.hexdigest()


def build_datasets():

    ensure_directories()

    stable_mask_train, transform, crs = build_stable_corn_mask_from_years(
        years=TRAIN_YEARS,
        aoi=NEBRASKA_BBOX
    )

    coords = generate_tile_coords(
        stable_mask_train,
        tile_size=TILE_SIZE,
        stride=STRIDE,
        max_tiles=MAX_TILES
    )

    coords_hash = hash_coords(coords)

    np.save(INTERIM_DIR / "tile_coords.npy", np.array(coords, dtype=np.int32))

    with open(INTERIM_DIR / "tile_hash.txt", "w") as f:
        f.write(coords_hash)

    X_train, y_train = [], []
    X_test, y_test = [], []

    for year in TRAIN_YEARS + [TEST_YEAR]:

        s2_path = SENTINEL_DIR / f"s2_ne_{year}.npy"

        if not s2_path.exists():
            raise RuntimeError(f"Sentinel data missing for {year}. Expected: {s2_path}")

        cube = np.load(s2_path)

        cube_clean = apply_scl_mask(cube)
        cube_spectral = cube_clean[..., :-1]

        ndvi = compute_ndvi(cube_spectral)
        ndre = compute_ndre(cube_spectral)

        tiles = extract_tiles_from_cube(
            cube_spectral, coords, tile_size=TILE_SIZE
        )

        ndre_tiles = extract_tiles_from_cube(
            ndre[..., None], coords, tile_size=TILE_SIZE
        )[..., 0]

        valid_ratio = np.mean(~np.isnan(tiles), axis=(1, 2, 3))
        keep = valid_ratio > 0.9  # require mostly clear pixels
        tiles = tiles[keep]
        ndre_tiles = ndre_tiles[keep]

        if tiles.size == 0:
            continue

        tiles = np.nan_to_num(tiles, nan=0.0)

        X = tiles.reshape(tiles.shape[0], -1)
        y = np.nanmean(ndre_tiles, axis=(1, 2))

        if year in TRAIN_YEARS:
            X_train.append(X)
            y_train.append(y)
        else:
            X_test.append(X)
            y_test.append(y)

    if not X_train or not X_test:
        raise RuntimeError("No valid tiles found after masking; check AOI, clouds, or tile_size.")

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    np.save(INTERIM_DIR / "X_train.npy", X_train)
    np.save(INTERIM_DIR / "y_train.npy", y_train)
    np.save(INTERIM_DIR / "X_test.npy", X_test)
    np.save(INTERIM_DIR / "y_test.npy", y_test)

    print("Dataset build complete.")
    print("Train samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])


if __name__ == "__main__":
    build_datasets()
