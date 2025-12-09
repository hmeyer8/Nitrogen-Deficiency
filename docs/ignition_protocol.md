Project Startup Guide — Nitrogen Deficiency Feature Pipeline

This document walks through the exact steps required to set up the project locally, download raw data, prepare training/testing datasets, and run the PCA vs Autoencoder experiment.

Follow each phase in order. Do not skip steps.

Phase 1 — Local Project Setup (One-Time)
1. Navigate to the Project Directory

macOS / Linux:
```
cd ~Nitrogen-Deficiency
```

Windows PowerShell:
```
Set-Location ~Nitrogen-Deficiency
```

2. Create and Activate a Virtual Environment

macOS / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You should now see:

(.venv)

Verify Python is active:

python --version

3. Install Project Dependencies

Upgrade pip and install all required packages (use the interpreter you just created so the right pip is invoked):

All platforms:
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If any package fails to install, stop here and resolve it before continuing.

4. Create the Local Secrets File

Create the .env file:

macOS / Linux:
```
nano .env
```

Windows PowerShell (choose any editor, e.g., Notepad):
```
notepad .env
```

Add your Copernicus Data Space credentials:

CDSE_CLIENT_ID=your_real_client_id
CDSE_CLIENT_SECRET=your_real_client_secret
GPU_ENABLED=false  # set to true to allow CUDA if available

The Sentinel Hub client is already pointed to Copernicus Data Space defaults. Override URLs only if you are using a different CDSE gateway.

Save the file and lock permissions:

macOS / Linux:
```
chmod 600 .env
```

Windows PowerShell:
```
icacls .env /inheritance:r
icacls .env /grant:r "${env:UserName}:(R,W)"
```
(These commands remove inherited permissions and grant the current user read/write access only.)

Verify that the environment variables load correctly:

macOS / Linux:
```
python - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
print(os.getenv("CDSE_CLIENT_ID") is not None)
PY
```

Windows PowerShell:
```
@'
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
print(os.getenv("CDSE_CLIENT_ID") is not None)
'@ | python
```

This must print:

True

Sanity check Copernicus auth with a tiny chip (fails fast on bad creds/scopes):
macOS / Linux:
```
python - <<'PY'
from datetime import date
from sentinelhub import BBox, CRS, SentinelHubRequest, MimeType, bbox_to_dimensions
from src.datasources.copernicus_client import get_sh_config, CDSE_S2_L2A

bbox = BBox(bbox=(-100.0, 41.0, -99.99, 41.01), crs=CRS.WGS84)  # tiny AOI
size = bbox_to_dimensions(bbox, resolution=60)
req = SentinelHubRequest(
    evalscript="return [B04,B08];",
    input_data=[SentinelHubRequest.input_data(
        data_collection=CDSE_S2_L2A,
        time_interval=(date(2023,6,1), date(2023,6,15)),
        mosaicking_order="leastCC"
    )],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=bbox,
    size=size,
    config=get_sh_config(),
)
arr = req.get_data()[0]
print(arr.shape)
PY
```

Windows PowerShell:
```
@'
from datetime import date
from sentinelhub import BBox, CRS, SentinelHubRequest, MimeType, bbox_to_dimensions
from src.datasources.copernicus_client import get_sh_config, CDSE_S2_L2A

bbox = BBox(bbox=(-100.0, 41.0, -99.99, 41.01), crs=CRS.WGS84)  # tiny AOI
size = bbox_to_dimensions(bbox, resolution=60)
req = SentinelHubRequest(
    evalscript="return [B04,B08];",
    input_data=[SentinelHubRequest.input_data(
        data_collection=CDSE_S2_L2A,
        time_interval=(date(2023,6,1), date(2023,6,15)),
        mosaicking_order="leastCC"
    )],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=bbox,
    size=size,
    config=get_sh_config(),
)
arr = req.get_data()[0]
print(arr.shape)
'@ | python
```
Expect a small shape (bands, h, w). Any 401/403 means credentials/scopes are wrong; fix before proceeding.

Phase 2 — Download Raw Data (Longest Step)
5. Download the USDA Cropland Data Layer (CDL)
```
python -m src.datasources.cdl_loader --years 2019 2020 2021 2022 2023 2024
```

This must create:

data/raw/cdl/

If the directory remains empty or an error occurs, stop and resolve it before continuing.

Sanity check CDL presence:
```
python - <<'PY'
from src.config import CDL_DIR
years = [2019,2020,2021,2022,2023,2024]
missing = [y for y in years if not (CDL_DIR / f"cdl_NE_{y}.tif").exists()]
print("Missing:", missing)
PY
```
Powershell:

```
@'
from src.config import CDL_DIR
years = [2019, 2020, 2021, 2022, 2023, 2024]
missing = [y for y in years if not (CDL_DIR / f"cdl_NE_{y}.tif").exists()]
print("Missing:", missing)
'@ | python -
```

`Missing: []` is required before moving on.

6. Download Sentinel-2 Data by Year (set `S2_SOURCE=cdse` for Copernicus Data Space, or `S2_SOURCE=pc` for free AWS/Planetary Computer COGs)

Run each command individually:
```
python -m src.datasources.sentinel_download --year 2019 --verbose
python -m src.datasources.sentinel_download --year 2020 --verbose
python -m src.datasources.sentinel_download --year 2021 --verbose
python -m src.datasources.sentinel_download --year 2022 --verbose
python -m src.datasources.sentinel_download --year 2023 --verbose
python -m src.datasources.sentinel_download --year 2024 --verbose
```
If you prefer an explicit Planetary Computer pull (bypassing sentinel_download toggle):
```
python -m src.datasources.sentinel_download_pc --year 2019 --verbose
...
python -m src.datasources.sentinel_download_pc --year 2024 --verbose
```

Each command must produce a time-series cube with 3 intra-season windows:

data/raw/sentinel/s2_ne_YYYY.npy

Do not proceed until all six years exist.

Sanity check Sentinel cubes:

linux
```
python - <<'PY'
import numpy as np
from src.config import SENTINEL_DIR
years = [2019,2020,2021,2022,2023,2024]
for y in years:
    p = SENTINEL_DIR / f"s2_ne_{y}.npy"
    try:
        arr = np.load(p)
        print(y, p.exists(), arr.shape)
    except Exception as e:
        print(y, "ERR", e)
PY
```
wsl
```
@'
import numpy as np
from src.config import SENTINEL_DIR
years = [2019, 2020, 2021, 2022, 2023, 2024]
for y in years:
    p = SENTINEL_DIR / f"s2_ne_{y}.npy"
    try:
        arr = np.load(p)
        print(f"{y}: exists={p.exists()} shape={arr.shape}")
    except Exception as e:
        print(f"{y}: ERR {e}")
'@ | python -
```

Each year should print `True (H, W, 11)`; fix any errors before continuing.

Phase 3 — Build Training and Test Datasets
7. Generate Leakage-Safe Train/Test Split (stable corn = intersection 2019–2024)
```
python -m src.experiments.prepare_dataset
```

Successful completion must print:

Dataset build complete.
Train samples: XXXXX
Test samples: XXXX
Tile coordinate hash saved: XXXXX

And create:

data/interim/
- X_train.npy
- y_train.npy
- X_test.npy
- y_test.npy
- tile_coords.npy
- tile_hash.txt

Sanity check tile counts (must be non-zero rows):
```
python - <<'PY'
import numpy as np
from src.config import INTERIM_DIR
for name in ["X_train","y_train","X_test","y_test"]:
    arr = np.load(INTERIM_DIR / f"{name}.npy")
    print(name, arr.shape)
PY
```
powershell
```
@'
import numpy as np
from src.config import INTERIM_DIR
for name in ["X_train", "y_train", "X_test", "y_test"]:
    arr = np.load(INTERIM_DIR / f"{name}.npy")
    print(f"{name}: {arr.shape}")
'@ | python -
```
If any shape shows 0 rows, reduce cloud threshold or shrink AOI and rerun prepare_dataset.

Phase 4 — Train Feature Models (PCA, Autoencoder, CatBoost)
8. Train Feature Models
```
python -m src.experiments.train_pca_ae_catboost
```

This must produce:

data/interim/
- scaler.joblib
- pca_model.joblib
- autoencoder.pt
- catboost_model.cbm
- Zp_train.npy / Zp_test.npy
- Za_train.npy / Za_test.npy
- y_pred_pca.npy
- y_pred_ae.npy
- y_pred_catboost.npy
- model_metrics.json
- pred_vs_true.png

If any error occurs here, review the stack trace before continuing.

Sanity check predictions exist:
```
python - <<'PY'
from src.config import INTERIM_DIR
for name in ["y_pred_pca.npy","y_pred_ae.npy","y_pred_catboost.npy"]:
    p = INTERIM_DIR / name
    print(name, p.exists())
PY
```
All should be `True`.

Phase 5 — Final Held-Out Evaluation (Most Important Result)
9. Evaluate Performance on 2024 Only
```
python -m src.experiments.evaluate_heldout_year
```

Expected output format:

[PCA / SVD]
R^2 = 0.4321
RMSE = 0.0874
Pearson r = 0.6712

[Autoencoder]
R^2 = 0.5123
RMSE = 0.0711
Pearson r = 0.7488

[CatBoost]
R^2 = 0.5012
RMSE = 0.0735
Pearson r = 0.7421

Interpretation:

If Autoencoder > PCA → nitrogen stress is likely nonlinear  
If PCA ≈ Autoencoder → nitrogen stress is primarily linear–spectral  
If CatBoost ≥ others → tree ensembles can capture needed interactions without explicit feature learning

These values represent the project’s final scientific conclusion.

Phase 6 — Visual Verification (Strongly Recommended)
10. Launch Jupyter for Visual Inspection

All platforms:
```
python -m pip install jupyterlab
jupyter lab
```

Run the following notebooks in order:

notebooks/01_explore_cdl.ipynb

notebooks/02_sample_tiles.ipynb

notebooks/03_feature_visualization.ipynb

If notebooks can’t import `src.*`, set the repo on your Python path before launching:
- Terminal: `export PYTHONPATH=$PWD && jupyter lab`
- Or add at the top of a notebook: `import sys, pathlib; sys.path.append(str(pathlib.Path.cwd()))`

These verify:

Crop-type stability

Tile alignment across years

PCA vs Autoencoder latent structure

What “Fully Complete” Means

When all phases succeed, the project has achieved:

True temporal generalization

Linear vs nonlinear nitrogen feature comparison

Cloud-masked spectral indices

Leakage-free held-out evaluation

Reproducible spatial sampling

Portfolio and publication-grade experimental rigor

At that point, this project is no longer a prototype — it is a complete scientific experiment.
