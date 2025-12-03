Project Startup Guide — Nitrogen Deficiency Feature Pipeline

This document walks through the exact steps required to set up the project locally, download all raw data, prepare training and testing datasets, and run the full PCA vs Autoencoder comparison experiment.

Follow each phase in order. Do not skip steps.

Phase 1 — Local Project Setup (One-Time)
1. Navigate to the Project Directory
cd ~/dev/nitrogen-features

2. Create and Activate a Virtual Environment
python3 -m venv .venv
source .venv/bin/activate


You should now see:

(.venv)


Verify Python is active:

python --version

3. Install Project Dependencies

Upgrade pip and install all required packages:

pip install --upgrade pip
pip install -r requirements.txt


If any package fails to install, stop here and resolve it before continuing.

4. Create the Local Secrets File

Create the .env file:

nano .env


Add your Copernicus Data Space credentials:

CDSE_CLIENT_ID=your_real_client_id
CDSE_CLIENT_SECRET=your_real_client_secret


Save the file and lock permissions:

chmod 600 .env


Verify that the environment variables load correctly:

python - <<EOF
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("CDSE_CLIENT_ID") is not None)
EOF


This must print:

True

Phase 2 — Download Raw Data (Longest Step)
5. Download the USDA Cropland Data Layer (CDL)
python -m src.datasources.cdl_loader


This must create:

data/raw/cdl/


If the directory remains empty or an error occurs, stop and resolve it before continuing.

6. Download Sentinel-2 Data by Year

Run each command individually:

python -m src.datasources.sentinel_download --year 2019
python -m src.datasources.sentinel_download --year 2020
python -m src.datasources.sentinel_download --year 2021
python -m src.datasources.sentinel_download --year 2022
python -m src.datasources.sentinel_download --year 2023
python -m src.datasources.sentinel_download --year 2024


Each command must produce:

data/raw/sentinel/s2_ne_YYYY.npy


Do not proceed until all six years exist.

Phase 3 — Build Training and Test Datasets
7. Generate Leakage-Safe Train/Test Split
python -m src.experiments.prepare_dataset


Successful completion must print:

Dataset build complete.
Train samples: XXXXX
Test samples: XXXX
Tile coordinate hash saved: XXXXX


And create:

data/interim/
├── X_train.npy
├── y_train.npy
├── X_test.npy
├── y_test.npy
├── tile_coords.npy
├── tile_hash.txt

Phase 4 — Train PCA vs Autoencoder Features
8. Train Feature Models
python -m src.experiments.train_pca_vs_ae


This must produce:

data/interim/
├── Zp_train.npy
├── Za_train.npy
├── y_pred_pca.npy
├── y_pred_ae.npy


If any error occurs here, review the stack trace before continuing.

Phase 5 — Final Held-Out Evaluation (Most Important Result)
9. Evaluate Performance on 2024 Only
python -m src.experiments.evaluate_heldout_year


Expected output format:

[PCA / SVD]
R² = 0.4321
RMSE = 0.0874
Pearson r = 0.6712

[Autoencoder]
R² = 0.5123
RMSE = 0.0711
Pearson r = 0.7488


Interpretation:

If Autoencoder > PCA → nitrogen stress is likely nonlinear

If PCA ≥ Autoencoder → nitrogen stress is primarily linear–spectral

These values represent the project’s final scientific conclusion.

Phase 6 — Visual Verification (Strongly Recommended)
10. Launch Jupyter for Visual Inspection
pip install jupyterlab
jupyter lab


Run the following notebooks in order:

notebooks/01_explore_cdl.ipynb

notebooks/02_sample_tiles.ipynb

notebooks/03_feature_visualization.ipynb

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

At that point, this project is no longer a prototype—it is a complete scientific experiment.