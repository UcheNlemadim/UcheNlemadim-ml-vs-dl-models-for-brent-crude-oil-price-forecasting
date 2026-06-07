"""Preprocessing utilities for the Brent crude oil univariate forecasting project.

This module provides a small, well-documented API used by notebooks and
`src/` modules to load configuration, read and clean the raw CSV, split the
series into train/validation/test partitions, and construct lag and rolling
features for a single-target (`Price`) time series.

Expectations and behaviour
- The config file at `configs/config.yaml` supplies all paths, date formats
    and hyperparameters used by these functions.
- Input data is the CSV `data/raw/Brent_Oil_Futures_Historical_Data_20002024.csv`
    (see config). Only the column named by `cfg['data']['target_column']` is
    retained; all other columns are dropped.
- All functions operate on a single-column `pd.DataFrame` indexed by
    `pd.DatetimeIndex`.

Provides
- `load_config(config_path: str) -> dict`
- `load_and_clean(cfg: dict) -> pd.DataFrame`
- `split_data(df: pd.DataFrame, cfg: dict) -> tuple`
- `create_lag_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame`
"""

import pathlib
from typing import Tuple

import pandas as pd
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config file and return as a dictionary.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    p = pathlib.Path(config_path)
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def load_and_clean(cfg: dict) -> pd.DataFrame:
    """Load raw CSV, keep only the target column and perform basic cleaning.

    Steps performed:
    - Read CSV from `cfg['data']['raw_path']`
    - Parse the date column using `cfg['data']['date_format']`
    - Set the Date column as the index and sort ascending
    - Keep only the target column named in `cfg['data']['target_column']`
    - Forward-fill NaN values
    - Drop duplicate index entries keeping the first

    Parameters
    ----------
    cfg : dict
        Configuration dictionary loaded from YAML.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame containing only the target column with a DatetimeIndex.
    """
    data_path = cfg["data"]["raw_path"]
    date_col = cfg["data"]["date_column"]
    target_col = cfg["data"]["target_column"]

    df = pd.read_csv(data_path, parse_dates=[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    # Keep only the target column
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    df = df[[target_col]].copy()

    original_len = len(df)

    # Forward-fill missing values
    df = df.ffill()

    # Remove duplicate index entries keeping the first
    deduped = df[~df.index.duplicated(keep="first")]
    removed = original_len - len(deduped)
    df = deduped

    # Print summary
    date_min = df.index.min()
    date_max = df.index.max()
    print(
        f"Total rows retained: {len(df)}; Date range: {date_min.date()} to {date_max.date()}; Duplicates removed: {removed}"
    )

    return df


def split_data(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the series DataFrame into train, validation and test sets by date.

    Uses the dates in `cfg['splits']` to perform the splits.

    Parameters
    ----------
    df : pd.DataFrame
        Single-column DataFrame indexed by date.
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    splits = cfg["splits"]
    train_end = pd.to_datetime(splits["train_end"]) if splits.get("train_end") else None
    val_end = pd.to_datetime(splits["val_end"]) if splits.get("val_end") else None
    test_start = pd.to_datetime(splits["test_start"]) if splits.get("test_start") else None

    if train_end is None or val_end is None or test_start is None:
        raise KeyError("Missing split date(s) in config")

    train = df.loc[:train_end]
    val = df.loc[(df.index > train_end) & (df.index <= val_end)]
    test = df.loc[test_start:]

    print(f"Train shape: {train.shape}; Val shape: {val.shape}; Test shape: {test.shape}")
    return train, val, test


def create_lag_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create lag, rolling mean and rolling std features for a univariate series.

    Parameters
    ----------
    df : pd.DataFrame
        Single-column DataFrame indexed by date.
    cfg : dict
        Configuration dictionary containing `features` settings.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the original target column and added feature columns.
    """
    target_col = cfg["data"]["target_column"]
    features_cfg = cfg.get("features", {})
    lags = features_cfg.get("lags", [])
    rolling_means = features_cfg.get("rolling_means", [])
    rolling_stds = features_cfg.get("rolling_stds", [])

    df_feat = df.copy()

    # Lag features
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)

    # Rolling means
    for window in rolling_means:
        df_feat[f"roll_mean_{window}"] = df_feat[target_col].rolling(window=window).mean()

    # Rolling stds
    for window in rolling_stds:
        df_feat[f"roll_std_{window}"] = df_feat[target_col].rolling(window=window).std()

    # Drop rows with NaNs introduced by lagging/rolling
    df_feat = df_feat.dropna()

    print(f"Feature DataFrame shape: {df_feat.shape}")
    return df_feat


if __name__ == "__main__":
    cfg = load_config()
    df = load_and_clean(cfg)
    print(df.head())
