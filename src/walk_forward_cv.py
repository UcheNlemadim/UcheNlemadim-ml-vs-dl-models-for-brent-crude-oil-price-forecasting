"""# Walk-forward cross-validation utilities

Simple helper to perform walk-forward (time series) cross-validation
using sklearn's `TimeSeriesSplit` and manual RMSE computation.
"""

import math

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def walk_forward_cv(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """Perform walk-forward cross-validation using TimeSeriesSplit.

    TimeSeriesSplit is used instead of standard KFold because it preserves
    temporal order: each fold's test observations come strictly after the
    training observations. This prevents leakage from future data into the
    model during training, which is essential for time series forecasting.

    Reference: Hyndman, R.J. and Athanasopoulos, G. (2018) Forecasting:
    Principles and Practice. 3rd ed. OTexts. (UWE Harvard style)

    Parameters
    ----------
    model : object
        A scikit-learn-like estimator implementing `fit(X, y)` and
        `predict(X)`.
    X : np.ndarray
        Feature matrix or 2D array for the estimator.
    y : np.ndarray
        Target array (1D) aligned with `X`.
    n_splits : int
        Number of time-series splits/folds.

    Returns
    -------
    dict
        Dictionary with keys:
        - "fold_rmse": list of float, RMSE per fold
        - "mean_rmse": float, mean RMSE across folds
        - "std_rmse": float, standard deviation of RMSE across folds

    Behaviour
    ---------
    For each split the model is trained on the training slice and evaluated
    on the subsequent test slice. RMSE is computed manually as
    sqrt(mean((y_true - y_pred)**2)). Fold info is printed for inspection.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    tss = TimeSeriesSplit(n_splits=n_splits)

    fold_rmse = []
    for i, (train_idx, test_idx) in enumerate(tss.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = np.mean((y_test - y_pred) ** 2)
        rmse = math.sqrt(mse)
        fold_rmse.append(float(rmse))

        print(f"Fold {i+1}: train_size={len(train_idx)}, test_size={len(test_idx)}, RMSE={rmse:.4f}")

    mean_rmse = float(np.mean(fold_rmse)) if fold_rmse else float('nan')
    std_rmse = float(np.std(fold_rmse, ddof=0)) if fold_rmse else float('nan')

    return {"fold_rmse": fold_rmse, "mean_rmse": mean_rmse, "std_rmse": std_rmse}
