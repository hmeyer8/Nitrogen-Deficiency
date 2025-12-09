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
STABLE_YEARS = TRAIN_YEARS + [TEST_YEAR]  # enforce corn-only pixels across all 5 years

TILE_SIZE = 32
STRIDE = 32
MAX_TILES = 2000
STABLE_CORN_THRESHOLD = 0.6
SEASON_WINDOWS = [
    (6, 1, 6, 30),
    (7, 1, 7, 31),
    (8, 1, 8, 31),
]


def resize_mask_to_cube(mask: np.ndarray, target_shape):
    """
    Simple nearest-neighbor resize of a mask to match the Sentinel cube grid.
    Avoids importing extra deps; good enough for stable corn mask.
    """
    target_h, target_w = target_shape
    src_h, src_w = mask.shape
    y_idx = np.floor(np.linspace(0, src_h - 1, target_h)).astype(int)
    x_idx = np.floor(np.linspace(0, src_w - 1, target_w)).astype(int)
    return mask[y_idx[:, None], x_idx]


def hash_coords(coords):
    h = hashlib.sha256()
    h.update(str(coords).encode())
    return h.hexdigest()


def build_datasets():

    ensure_directories()

    # Enforce corn-only tiles across 2019–2024 by intersecting CDL masks
    stable_mask_train, transform, crs = build_stable_corn_mask_from_years(
        years=STABLE_YEARS,
        aoi=NEBRASKA_BBOX
    )

    sample_cube = np.load(SENTINEL_DIR / f"s2_ne_{TRAIN_YEARS[0]}.npy")
    if sample_cube.ndim == 4:
        sample_cube = sample_cube[0]  # time dimension present
    stable_mask_train = resize_mask_to_cube(
        stable_mask_train,
        target_shape=sample_cube.shape[:2]
    )

    coords = generate_tile_coords(
        stable_mask_train,
        tile_size=TILE_SIZE,
        stride=STRIDE,
        max_tiles=MAX_TILES,
        stable_threshold=STABLE_CORN_THRESHOLD,
    )

    coords_hash = hash_coords(coords)

    np.save(INTERIM_DIR / "tile_coords.npy", np.array(coords, dtype=np.int32))

    with open(INTERIM_DIR / "tile_hash.txt", "w") as f:
        f.write(coords_hash)

    X_train, y_train = [], []
    X_test, y_test = [], []
    ndre_ts_train, ndre_ts_test = [], []
    ndvi_ts_train, ndvi_ts_test = [], []

    for year in TRAIN_YEARS + [TEST_YEAR]:

        s2_path = SENTINEL_DIR / f"s2_ne_{year}.npy"

        if not s2_path.exists():
            raise RuntimeError(f"Sentinel data missing for {year}. Expected: {s2_path}")

        cube = np.load(s2_path)
        if cube.ndim == 3:
            cube = cube[None, ...]  # add time dimension if single mosaic

        season_tiles = []
        season_ndre_tiles = []
        season_ndvi_tiles = []
        for t in range(cube.shape[0]):
            cube_t = cube[t]
            cube_clean = apply_scl_mask(cube_t)
            cube_spectral = cube_clean[..., :-1]

            ndre = compute_ndre(cube_spectral)
            ndvi = compute_ndvi(cube_spectral)

            tiles_t = extract_tiles_from_cube(
                cube_spectral, coords, tile_size=TILE_SIZE
            )

            ndre_tiles_t = extract_tiles_from_cube(
                ndre[..., None], coords, tile_size=TILE_SIZE
            )[..., 0]

            ndvi_tiles_t = extract_tiles_from_cube(
                ndvi[..., None], coords, tile_size=TILE_SIZE
            )[..., 0]

            season_tiles.append(tiles_t)
            season_ndre_tiles.append(ndre_tiles_t)
            season_ndvi_tiles.append(ndvi_tiles_t)

        tiles = np.stack(season_tiles, axis=1)  # (num_tiles, T, H, W, B)
        ndre_tiles = np.stack(season_ndre_tiles, axis=1)  # (num_tiles, T, H, W)
        ndvi_tiles = np.stack(season_ndvi_tiles, axis=1)  # (num_tiles, T, H, W)

        valid_ratio = np.mean(~np.isnan(tiles), axis=(1, 2, 3, 4))
        keep = valid_ratio > 0.9  # require mostly clear pixels
        tiles = tiles[keep]
        ndre_tiles = ndre_tiles[keep]
        ndvi_tiles = ndvi_tiles[keep]

        if tiles.size == 0:
            continue

        tiles = np.nan_to_num(tiles, nan=0.0)

        # Collapse time and space into feature vectors; time axis retained by flattening
        X = tiles.reshape(tiles.shape[0], -1)
        # Nitrogen deficiency proxy: worst NDRE across the season (lower = more deficient)
        ndre_tile_means = np.nanmean(ndre_tiles, axis=(2, 3))  # (num_tiles, T)
        ndvi_tile_means = np.nanmean(ndvi_tiles, axis=(2, 3))  # (num_tiles, T)
        y = np.nanmin(ndre_tile_means, axis=1)

        if year in TRAIN_YEARS:
            X_train.append(X)
            y_train.append(y)
            ndre_ts_train.append(ndre_tile_means)
            ndvi_ts_train.append(ndvi_tile_means)
        else:
            X_test.append(X)
            y_test.append(y)
            ndre_ts_test.append(ndre_tile_means)
            ndvi_ts_test.append(ndvi_tile_means)

    if not X_train or not X_test:
        raise RuntimeError("No valid tiles found after masking; check AOI, clouds, or tile_size.")

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    ndre_ts_train = np.concatenate(ndre_ts_train, axis=0)
    ndre_ts_test = np.concatenate(ndre_ts_test, axis=0)
    ndvi_ts_train = np.concatenate(ndvi_ts_train, axis=0)
    ndvi_ts_test = np.concatenate(ndvi_ts_test, axis=0)

    # Time-series targets for forecasting end-of-season NDRE
    y_future_train = ndre_ts_train[:, -1]  # last window NDRE mean
    y_future_test = ndre_ts_test[:, -1]
    y_future_mean, y_future_std = y_future_train.mean(), y_future_train.std()
    y_future_train_deficit_score = (y_future_train - y_future_mean) / (y_future_std + 1e-8)
    y_future_test_deficit_score = (y_future_test - y_future_mean) / (y_future_std + 1e-8)
    future_deficit_thresh = np.percentile(y_future_train, 25)
    y_future_train_deficit_label = (y_future_train < future_deficit_thresh).astype(np.int8)
    y_future_test_deficit_label = (y_future_test < future_deficit_thresh).astype(np.int8)

    # Nitrogen deficiency proxy: NDRE z-score and low-quantile flag (train-driven)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_deficit_score = (y_train - y_mean) / (y_std + 1e-8)
    y_test_deficit_score = (y_test - y_mean) / (y_std + 1e-8)
    deficit_thresh = np.percentile(y_train, 25)  # bottom quartile = likely N-deficient
    y_train_deficit_label = (y_train < deficit_thresh).astype(np.int8)
    y_test_deficit_label = (y_test < deficit_thresh).astype(np.int8)

    np.save(INTERIM_DIR / "X_train.npy", X_train)
    np.save(INTERIM_DIR / "y_train.npy", y_train)
    np.save(INTERIM_DIR / "X_test.npy", X_test)
    np.save(INTERIM_DIR / "y_test.npy", y_test)
    np.save(INTERIM_DIR / "ndre_ts_train.npy", ndre_ts_train)
    np.save(INTERIM_DIR / "ndre_ts_test.npy", ndre_ts_test)
    np.save(INTERIM_DIR / "ndvi_ts_train.npy", ndvi_ts_train)
    np.save(INTERIM_DIR / "ndvi_ts_test.npy", ndvi_ts_test)
    np.save(INTERIM_DIR / "y_train_deficit_score.npy", y_train_deficit_score)
    np.save(INTERIM_DIR / "y_test_deficit_score.npy", y_test_deficit_score)
    np.save(INTERIM_DIR / "y_train_deficit_label.npy", y_train_deficit_label)
    np.save(INTERIM_DIR / "y_test_deficit_label.npy", y_test_deficit_label)
    np.save(INTERIM_DIR / "y_future_train.npy", y_future_train)
    np.save(INTERIM_DIR / "y_future_test.npy", y_future_test)
    np.save(INTERIM_DIR / "y_future_train_deficit_score.npy", y_future_train_deficit_score)
    np.save(INTERIM_DIR / "y_future_test_deficit_score.npy", y_future_test_deficit_score)
    np.save(INTERIM_DIR / "y_future_train_deficit_label.npy", y_future_train_deficit_label)
    np.save(INTERIM_DIR / "y_future_test_deficit_label.npy", y_future_test_deficit_label)

    with open(INTERIM_DIR / "deficit_threshold.txt", "w") as f:
        f.write(f"train_mean={y_mean}\ntrain_std={y_std}\nq25={deficit_thresh}\n")
    with open(INTERIM_DIR / "future_deficit_threshold.txt", "w") as f:
        f.write(f"train_mean={y_future_mean}\ntrain_std={y_future_std}\nq25={future_deficit_thresh}\n")

    print("Dataset build complete.")
    print("Train samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])


if __name__ == "__main__":
    build_datasets()
