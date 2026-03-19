"""
loaders.py
----------
Data loading utilities for fat-tail statistical modeling.

Provides functions to load data from common file formats (CSV, Parquet, NumPy)
and to perform basic data validation before analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def load_csv(
    filepath: str | Path,
    column: str | None = None,
    dtype: type = float,
    dropna: bool = True,
) -> NDArray[np.float64]:
    """Load data from a CSV file and return as a NumPy array.

    Args:
        filepath: Path to the CSV file.
        column: Column name to extract. If None, expects a single-column file
            or returns the first numeric column.
        dtype: Data type to cast values to.
        dropna: If True, drop NaN values before returning.

    Returns:
        1-D NumPy array of float64 values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the specified column is not found.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if column is not None:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
        series = df[column]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in the CSV file.")
        series = df[numeric_cols[0]]

    if dropna:
        series = series.dropna()

    return series.to_numpy(dtype=dtype)


def load_parquet(
    filepath: str | Path,
    column: str | None = None,
    dropna: bool = True,
) -> NDArray[np.float64]:
    """Load data from a Parquet file and return as a NumPy array.

    Args:
        filepath: Path to the Parquet file.
        column: Column name to extract. If None, uses the first numeric column.
        dropna: If True, drop NaN values before returning.

    Returns:
        1-D NumPy array of float64 values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the specified column is not found.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_parquet(path)

    if column is not None:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
        series = df[column]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in the Parquet file.")
        series = df[numeric_cols[0]]

    if dropna:
        series = series.dropna()

    return series.to_numpy(dtype=np.float64)


def load_numpy(filepath: str | Path) -> NDArray[np.float64]:
    """Load data from a NumPy .npy or .npz file.

    For .npz files, loads the first array found.

    Args:
        filepath: Path to the .npy or .npz file.

    Returns:
        1-D NumPy array of float64 values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the array cannot be interpreted as 1-D float data.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".npz":
        data = np.load(path)
        first_key = list(data.keys())[0]
        arr = data[first_key]
    else:
        arr = np.load(path)

    arr = arr.flatten().astype(np.float64)
    return arr


def validate_data(
    data: NDArray[np.float64],
    allow_nan: bool = False,
    allow_inf: bool = False,
    min_samples: int = 10,
) -> NDArray[np.float64]:
    """Validate and clean a NumPy array for statistical analysis.

    Args:
        data: Input array to validate.
        allow_nan: If False, raises an error if NaN values are present.
        allow_inf: If False, raises an error if Inf values are present.
        min_samples: Minimum number of valid samples required.

    Returns:
        Cleaned and validated 1-D NumPy array.

    Raises:
        ValueError: If data fails validation checks.
    """
    data = np.asarray(data, dtype=np.float64).flatten()

    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))

    if nan_count > 0:
        if not allow_nan:
            raise ValueError(f"Data contains {nan_count} NaN value(s). Set allow_nan=True to drop them.")
        data = data[~np.isnan(data)]

    if inf_count > 0:
        if not allow_inf:
            raise ValueError(f"Data contains {inf_count} Inf value(s). Set allow_inf=True to drop them.")
        data = data[~np.isinf(data)]

    if len(data) < min_samples:
        raise ValueError(
            f"Insufficient samples after cleaning: {len(data)} < {min_samples} required."
        )

    return data


def load_dataframe(
    filepath: str | Path,
    columns: list[str] | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Load a CSV or Parquet file as a DataFrame.

    Automatically detects file format from extension.

    Args:
        filepath: Path to CSV or Parquet file.
        columns: List of column names to load. If None, loads all columns.
        dropna: If True, drops rows with any NaN in the selected columns.

    Returns:
        Pandas DataFrame with selected columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Use .csv or .parquet.")

    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        df = df[columns]

    if dropna:
        df = df.dropna()

    return df.reset_index(drop=True)
