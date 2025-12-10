# src/experiments/prepare_dataset.py

import hashlib
import os
import numpy as np
from pathlib import Path

from src.config import INTERIM_DIR, SENTINEL_DIR
from src.datasources.cdl_loader import build_stable_corn_mask_from_years
from src.geo.tiling import generate_tile_coords
from src.features.indices import compute_ndvi, compute_ndre
from src.utils.io import ensure_directories
from src.geo.aoi_nebraska import NEBRASKA_BBOX


TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]
TEST_YEAR = 2024
STABLE_YEARS = TRAIN_YEARS + [TEST_YEAR]  # enforce corn-only pixels across all 5 years

TILE_SIZE = 32
STRIDE = 32
# Allow tuning via env without editing code; defaults favor accuracy while still fitting 32 GB RAM.
MAX_TILES = int(os.getenv("MAX_TILES", "800"))
MIN_CLEAR_RATIO = float(os.getenv("MIN_CLEAR_RATIO", "0.2"))
STABLE_CORN_THRESHOLD = float(os.getenv("STABLE_THRESHOLD", "0.2"))
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


def extract_tile_time_series(cube: np.memmap, coords, tile_size: int, min_clear_ratio: float):
    """
    Memory-friendly extraction: operate tile-by-tile instead of computing indices
    on the full 8k x 16k frame.
    Returns lists of tiles and index time series.
    """
    tiles, ndre_tiles, ndvi_tiles = [], [], []
    T = cube.shape[0]

    for y, x in coords:
        tile_stack = []
        ndre_stack = []
        ndvi_stack = []

        for t in range(T):
            tile = cube[t, y:y + tile_size, x:x + tile_size, :]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                break  # skip partial tiles on borders

            spectral = tile[..., :-1].astype(np.float32, copy=True)
            scl = tile[..., -1]
            invalid = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 11)
            spectral[invalid] = np.nan

            ndre = compute_ndre(spectral)
            ndvi = compute_ndvi(spectral)

            tile_stack.append(spectral)
            ndre_stack.append(ndre)
            ndvi_stack.append(ndvi)

        if len(tile_stack) != T:
            continue

        tile_stack = np.stack(tile_stack, axis=0)
        ndre_stack = np.stack(ndre_stack, axis=0)
        ndvi_stack = np.stack(ndvi_stack, axis=0)

        valid_ratio = np.mean(~np.isnan(tile_stack))
        if valid_ratio < min_clear_ratio:
            continue
        # Require NDRE to have at least some finite pixels
        if not np.isfinite(ndre_stack).any():
            continue

        tiles.append(tile_stack)
        ndre_tiles.append(ndre_stack)
        ndvi_tiles.append(ndvi_stack)

    if not tiles:
        return (
            np.empty((0, T, tile_size, tile_size, cube.shape[-1] - 1), dtype=np.float32),
            np.empty((0, T, tile_size, tile_size), dtype=np.float32),
            np.empty((0, T, tile_size, tile_size), dtype=np.float32),
        )

    return (
        np.stack(tiles, axis=0).astype(np.float32),
        np.stack(ndre_tiles, axis=0).astype(np.float32),
        np.stack(ndvi_tiles, axis=0).astype(np.float32),
    )


def build_datasets():

    ensure_directories()

    # Enforce corn-only tiles across 2019–2024 by intersecting CDL masks
    stable_mask_train, transform, crs = build_stable_corn_mask_from_years(
        years=STABLE_YEARS,
        aoi=NEBRASKA_BBOX
    )

    sample_cube = np.load(SENTINEL_DIR / f"s2_ne_{TRAIN_YEARS[0]}.npy", mmap_mode="r")
    if sample_cube.ndim == 4:
        sample_cube = sample_cube[0]  # time dimension present
    stable_mask_train = resize_mask_to_cube(
        stable_mask_train,
        target_shape=sample_cube.shape[:2]
    )
    coverage_union = np.zeros(sample_cube.shape[:2], dtype=bool)
    for y in STABLE_YEARS:
        s2_path = SENTINEL_DIR / f"s2_ne_{y}.npy"
        if not s2_path.exists():
            continue
        cube_y = np.load(s2_path, mmap_mode="r")
        cube_slice = cube_y[0] if cube_y.ndim == 4 else cube_y
        coverage_union |= np.isfinite(cube_slice[..., 0])
        del cube_y, cube_slice
    stable_mask_train = stable_mask_train * coverage_union

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

        cube = np.load(s2_path, mmap_mode="r")  # memory map to avoid loading full stack into RAM
        if cube.ndim == 3:
            cube = cube[None, ...]  # add time dimension if single mosaic

        tiles, ndre_tiles, ndvi_tiles = extract_tile_time_series(
            cube=cube,
            coords=coords,
            tile_size=TILE_SIZE,
            min_clear_ratio=MIN_CLEAR_RATIO,
        )

        if tiles.size == 0:
            print(f"[prepare_dataset] Year {year}: no tiles passed clear_ratio={MIN_CLEAR_RATIO}")
            continue

        print(f"[prepare_dataset] Year {year}: tiles kept {tiles.shape[0]}")

        tiles = np.nan_to_num(tiles, nan=0.0)

        # Collapse time and space into feature vectors; time axis retained by flattening
        X = tiles.reshape(tiles.shape[0], -1)
        # Nitrogen deficiency proxy: worst NDRE across the season (lower = more deficient)
        ndre_tile_means = np.nanmean(ndre_tiles, axis=(2, 3))  # (num_tiles, T)
        ndvi_tile_means = np.nanmean(ndvi_tiles, axis=(2, 3))  # (num_tiles, T)

        def fill_time_nans(arr):
            # arr: (N, T); replace NaNs in each row with that row's finite mean
            filled = arr.copy()
            row_mean = np.nanmean(filled, axis=1)
            for i in range(filled.shape[0]):
                if np.isnan(row_mean[i]):
                    continue
                nan_mask = ~np.isfinite(filled[i])
                filled[i, nan_mask] = row_mean[i]
            return filled

        # Keep tiles with at least one finite NDRE time step
        keep_mask = np.any(np.isfinite(ndre_tile_means), axis=1)
        X = X[keep_mask]
        ndre_tile_means = ndre_tile_means[keep_mask]
        ndvi_tile_means = ndvi_tile_means[keep_mask]

        if X.size == 0:
            print(f"[prepare_dataset] Year {year}: all tiles dropped due to NaNs in NDRE means.")
            continue

        ndre_tile_means = fill_time_nans(ndre_tile_means)
        ndvi_tile_means = fill_time_nans(ndvi_tile_means)

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

        # Free per-year arrays before next iteration
        del cube, tiles, ndre_tiles, ndvi_tiles, ndre_tile_means, ndvi_tile_means

    if not X_train or not X_test:
        raise RuntimeError("No valid tiles found after masking; check AOI, clouds, or tile_size.")

    X_train = np.concatenate(X_train, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train, axis=0).astype(np.float32)
    X_test = np.concatenate(X_test, axis=0).astype(np.float32)
    y_test = np.concatenate(y_test, axis=0).astype(np.float32)
    ndre_ts_train = np.concatenate(ndre_ts_train, axis=0).astype(np.float32)
    ndre_ts_test = np.concatenate(ndre_ts_test, axis=0).astype(np.float32)
    ndvi_ts_train = np.concatenate(ndvi_ts_train, axis=0).astype(np.float32)
    ndvi_ts_test = np.concatenate(ndvi_ts_test, axis=0).astype(np.float32)

    # Fallback: if test set too small, peel a few samples from train for evaluation
    if X_test.shape[0] < 5 and X_train.shape[0] > 20:
        k = min(20, X_train.shape[0] // 5)
        X_train, X_extra = X_train[:-k], X_train[-k:]
        y_train, y_extra = y_train[:-k], y_train[-k:]
        ndre_ts_train, ndre_ts_extra = ndre_ts_train[:-k], ndre_ts_train[-k:]
        ndvi_ts_train, ndvi_ts_extra = ndvi_ts_train[:-k], ndvi_ts_train[-k:]

        X_test = np.concatenate([X_test, X_extra], axis=0)
        y_test = np.concatenate([y_test, y_extra], axis=0)
        ndre_ts_test = np.concatenate([ndre_ts_test, ndre_ts_extra], axis=0)
        ndvi_ts_test = np.concatenate([ndvi_ts_test, ndvi_ts_extra], axis=0)

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
