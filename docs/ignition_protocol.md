Project Startup Guide - Hybrid Temporal SVD Pipeline

This guide shows how to set up locally, download data, build the time-series datasets, and train the hybrid nitrogen-risk model (temporal SVD + CatBoost + AE + fusion). Follow `docs/eval_protocol.md` for the fixed 2019–2020 train, 2021–2022 val, 2023–2024 test split.

Phase 1 - Local Project Setup (one-time)
1) Navigate to repo:
   - macOS/Linux: `cd ~/Nitrogen-Deficiency`
   - Windows PowerShell: `Set-Location ~/Nitrogen-Deficiency`
2) Create + activate venv:
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows: `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
3) Install deps: `python -m pip install --upgrade pip && python -m pip install -r requirements.txt`
4) Create `.env` with Copernicus creds:
   ```
   CDSE_CLIENT_ID=your_real_client_id
   CDSE_CLIENT_SECRET=your_real_client_secret
   GPU_ENABLED=false
   ```
   (Set `GPU_ENABLED=true` only if CUDA is available.)
5) Quick auth sanity check (tiny chip):
   - macOS/Linux: run the Python block in README or sentinel sanity check.
   - Windows: same block via PowerShell heredoc.

Phase 2 - Download Raw Data
1) CDL (required):
   ```
   python -m src.datasources.cdl_loader --years 2019 2020 2021 2022 2023 2024
   ```
   Verify all tif files exist in `data/raw/cdl/`.
2) Sentinel-2 L2A mosaics (3 phenology windows/year, NE AOI):
   - Set crop (defines AOI via CDL): `TARGET_CROP=corn`
   - Choose source:
     - `S2_SOURCE=cdse` (default Copernicus Data Space via sentinelhub)
     - or `S2_SOURCE=pc` (Planetary Computer/AWS COGs)
   - Download per year:
     ```
     python -m src.datasources.sentinel_download --year 2019 --verbose
     ...
     python -m src.datasources.sentinel_download --year 2024 --verbose
     ```
     (Or directly call `python -m src.datasources.sentinel_download_pc --year YYYY --verbose` for PC.)
   Ensure `data/raw/sentinel/s2_ne_YYYY.npy` exists for all years.

Phase 3 - Build Time-Series Datasets (stable corn only)
```
python -m src.experiments.prepare_dataset
```
Outputs in `data/interim/`:
- ndre_ts_train.npy / ndre_ts_test.npy (N x 5 NDRE series)
- ndvi_ts_train.npy / ndvi_ts_test.npy
- y_min_train.npy / y_min_test.npy (NDRE minima)
- y_train_deficit_label.npy / y_test_deficit_label.npy (bottom-quantile flags)
- y_train_deficit_score.npy / y_test_deficit_score.npy (z-scores)
- deficit_threshold.txt, tile_coords.npy, tile_hash.txt
Quick check:
```
python - <<'PY'
import numpy as np
from src.config import INTERIM_DIR
for name in ["ndre_ts_train","ndre_ts_test","y_train_deficit_label","y_test_deficit_label"]:
    arr = np.load(INTERIM_DIR / f"{name}.npy")
    print(name, arr.shape)
PY
```

Phase 4 - Train Hybrid Phenology Model
```
# Optional fusion weights: RISK_ALPHA / RISK_BETA / RISK_GAMMA
python -m src.experiments.train_temporal_hybrid
```
Artifacts (data/interim):
- SVD: svd_components.npy, svd_mean.npy, svd_std.npy, svd_meta.json, svd_scores_{split}.npy, svd_residual_norm_{split}.npy
- CatBoost: catboost_classifier.cbm, catboost_prob_{split}.npy
- Autoencoder: temporal_ae.pt, ae_anomaly_{split}.npy
- Fusion: hybrid_risk_{split}.npy, hybrid_threshold.txt, hybrid_metrics.json

Phase 5 - Optional Extras
- Early-forecast GRU (early NDRE -> end-season NDRE/deficit):
  ```
  python -m src.experiments.train_temporal_forecaster
  ```
- Time-series diagnostics (mean trajectories, derivative outliers):
  ```
  python -m src.experiments.time_series_diagnostics
  ```

Phase 6 - Visual Verification (recommended)
1) Launch Jupyter: `python -m pip install jupyterlab && jupyter lab`
2) Notebooks:
   - notebooks/01_explore_cdl.ipynb
   - notebooks/02_sample_tiles.ipynb
   - notebooks/03_feature_visualization.ipynb (SVD loadings, CatBoost importance, AE recon error)
3) PYTHONPATH if imports fail:
   - `export PYTHONPATH=$PWD && jupyter lab`
   - or prepend in notebook: `import sys, pathlib; sys.path.append(str(pathlib.Path.cwd()))`

Done when:
- Stable-corn mask and Sentinel cubes verified.
- `prepare_dataset` produced non-empty NDRE/NDVI time series and labels.
- `train_temporal_hybrid` finished and wrote `hybrid_metrics.json` + risk scores.
- Optional diagnostics/forecaster run without errors.
