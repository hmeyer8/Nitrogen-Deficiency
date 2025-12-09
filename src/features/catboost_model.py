# src/features/catboost_model.py
import numpy as np
from typing import Optional

try:
    from catboost import CatBoostRegressor, Pool
except ImportError as e:
    raise ImportError("catboost is required for gradient boosting; pip install catboost") from e


def train_catboost_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Optional[np.ndarray] = None,
    y_valid: Optional[np.ndarray] = None,
    iterations: int = 400,
    depth: int = 6,
    learning_rate: float = 0.05,
    random_seed: int = 0,
    use_gpu: bool = True,
    max_bin: int = 64,
    subsample: Optional[float] = 0.8,
    rsm: Optional[float] = None,
    gpu_ram_part: float = 0.6,
    bootstrap_type: str = "Poisson",
    bagging_temperature: Optional[float] = None,
):
    """
    Train a CatBoost regressor on standardized pixel vectors.
    Uses early stopping when validation data is provided.
    """
    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_valid, y_valid) if X_valid is not None and y_valid is not None else None

    params = dict(
        loss_function="RMSE",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        eval_metric="RMSE",
        random_seed=random_seed,
        verbose=100,
        max_bin=max_bin,
        task_type="GPU" if use_gpu else "CPU",
        bootstrap_type=bootstrap_type,
    )

    if use_gpu:
        params["devices"] = "0"
        params["gpu_ram_part"] = gpu_ram_part  # limit VRAM fraction
    if subsample is not None:
        params["subsample"] = subsample
    if rsm is not None:
        params["rsm"] = rsm
    if bagging_temperature is not None:
        params["bagging_temperature"] = bagging_temperature
    if eval_pool is not None:
        params["od_type"] = "Iter"
        params["od_wait"] = 50

    model = CatBoostRegressor(**params)

    model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)
    return model


def predict_catboost(model: CatBoostRegressor, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def load_catboost_model(path) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(path)
    return model
