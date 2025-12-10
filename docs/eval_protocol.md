# Evaluation Protocol for Nitrogen Temporal Models

This document fixes a reproducible protocol so reported metrics are scientifically defensible and comparable across runs.

## Fixed dataset splits (locked)
- **Train:** 2019–2020
- **Validation:** 2021–2022 (used for all tuning and threshold selection)
- **Test:** 2023–2024 (never used for tuning; reported once per configuration)
- Use the same phenology windows and AOI for all splits.

## Label definition (NDRE proxy)
- Use `NDRE_DEFICIT_Q=50` as the default quantile for deficit labels.
- A label scheme is valid only if each split has both classes with rates between 5–95%. If not, adjust the quantile and rerun from scratch; do **not** tune quantile based on test metrics.
- Record the chosen quantile, threshold, and class rates from `data/interim/deficit_threshold.txt` with every run.

## Run procedure (per experiment)
1) Set env once, e.g.:
   - `NDRE_DEFICIT_Q=50`
   - `MAX_TILES=2000`, `MIN_CLEAR_RATIO=0.1`, `STABLE_THRESHOLD=0.2`
   - `TARGET_MODE=deficit_score` (or `ndre` if explicitly testing raw NDRE)
2) Rebuild and train:
   ```
   rm -rf data/interim
   python -m src.experiments.prepare_dataset
   python -m src.experiments.train_temporal_hybrid
   python -m src.experiments.time_series_diagnostics
   ```
3) Do not modify the test split or windows during tuning. Use only validation metrics for hyperparameter decisions.

## Reporting
- Primary metrics: AUC and Average Precision on **validation** and **test** for both CatBoost and Hybrid (from `data/interim/hybrid_metrics.json`).
- Secondary: class rates (train/val/test), SVD rank/EVR, CatBoost iterations used, AE train recon MSE.
- Threshold `tau`: choose on validation (already computed); apply unchanged to test.
- If reporting a single “headline” number, use Hybrid AUC on the fixed test split, accompanied by validation AUC/AP and class rates.

## Reproducibility
- Record: git commit hash, `.env` contents (or overridden env vars), `deficit_threshold.txt`, and `hybrid_metrics.json`.
- If randomness is involved, run 3 seeds and report mean ± std on validation; use the best seed’s trained model to score the fixed test once.

## Guardrails against overfitting
- Never retune on the test split; if the test metric is used, freeze code/config first.
- Do not rewindow or redownload test years during tuning.
- Avoid quantile/threshold adjustments based on test behavior.
- Keep a single canonical split; if experimenting with alternative splits, treat them as separate experiments and do not cherry-pick.
