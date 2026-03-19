"""
helpers.py
----------
Shared helper functions for the fat-tail package.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ensure_numpy(data: Any) -> NDArray[np.float64]:
    """Ensure input is a 1-D float64 NumPy array."""
    return np.asarray(data, dtype=np.float64).flatten()


def descriptive_stats(data: NDArray[np.float64]) -> dict[str, float]:
    """Compute basic descriptive statistics."""
    data = ensure_numpy(data)
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
        "count": float(len(data)),
    }
