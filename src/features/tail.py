"""
features/tail.py
-----------------
Trích xuất và phân tích đặc trưng đuôi phân phối (tail risk features):
  - Empirical VaR / CVaR (Expected Shortfall)
  - Hill estimator (tail index)
  - Tail ratio
  - Exceedance / POT threshold selection
  - Tail probability & quantile mapping
  - Tail feature summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Tuple

ArrayLike = Union[np.ndarray, pd.Series]


# ---------------------------------------------------------------------------
# Empirical VaR / CVaR
# ---------------------------------------------------------------------------

def empirical_var(
    returns: ArrayLike,
    confidence: float = 0.95,
    side: str = "left",
) -> float:
    """
    Tính Value-at-Risk theo phương pháp lịch sử (non-parametric).

    Parameters
    ----------
    returns    : chuỗi return
    confidence : mức tin cậy (default 0.95 → VaR 95%)
    side       : 'left'  → đuôi trái (loss), trả về giá trị âm
                 'right' → đuôi phải

    Returns
    -------
    float – VaR (giá trị âm với side='left')
    """
    r = pd.Series(returns).dropna().values
    alpha = 1 - confidence if side == "left" else confidence
    return float(np.quantile(r, alpha))


def empirical_cvar(
    returns: ArrayLike,
    confidence: float = 0.95,
    side: str = "left",
) -> float:
    """
    Tính Conditional VaR (Expected Shortfall) lịch sử.

    CVaR = E[R | R ≤ VaR]  (với side='left')

    Parameters
    ----------
    returns    : chuỗi return
    confidence : mức tin cậy
    side       : 'left' (loss tail) hoặc 'right'

    Returns
    -------
    float – CVaR (giá trị âm với side='left')
    """
    r = pd.Series(returns).dropna().values
    var = empirical_var(r, confidence=confidence, side=side)
    if side == "left":
        tail = r[r <= var]
    else:
        tail = r[r >= var]
    return float(tail.mean()) if len(tail) > 0 else np.nan


# ---------------------------------------------------------------------------
# Hill Estimator – Tail Index
# ---------------------------------------------------------------------------

def hill_estimator(
    returns: ArrayLike,
    k: int | None = None,
    k_fraction: float = 0.1,
) -> float:
    """
    Ước lượng tail index (α) bằng Hill estimator.

    Tail index α càng nhỏ → đuôi càng dày (fat-tail).
    Phân phối chuẩn: α → ∞.

    Parameters
    ----------
    returns    : chuỗi return (sẽ lấy giá trị tuyệt đối của phần âm)
    k          : số quan sát đuôi dùng để ước lượng.
                 Nếu None → dùng k_fraction * n
    k_fraction : tỷ lệ đuôi (default 10%)

    Returns
    -------
    float – ước lượng tail index α (> 0)

    References
    ----------
    Hill, B. M. (1975). A simple general approach to inference about
    the tail of a distribution. Ann. Statist., 3(5), 1163–1174.
    """
    r = pd.Series(returns).dropna()
    losses = np.sort(np.abs(r[r < 0].values))[::-1]  # giảm dần
    n = len(losses)
    if n < 10:
        return np.nan
    if k is None:
        k = max(2, int(k_fraction * n))
    k = min(k, n - 1)
    log_ratios = np.log(losses[:k]) - np.log(losses[k])
    alpha_inv = np.mean(log_ratios)  # 1/α
    return float(1.0 / alpha_inv) if alpha_inv > 0 else np.nan


def hill_plot(
    returns: ArrayLike,
    k_max_fraction: float = 0.3,
) -> pd.DataFrame:
    """
    Tính Hill estimator cho nhiều giá trị k (để vẽ Hill plot).

    Returns
    -------
    pd.DataFrame với cột ['k', 'tail_index']
    """
    r = pd.Series(returns).dropna()
    losses = np.sort(np.abs(r[r < 0].values))[::-1]
    n = len(losses)
    k_max = max(5, int(k_max_fraction * n))
    results = []
    for k in range(2, k_max + 1):
        alpha = hill_estimator(r, k=k)
        results.append({"k": k, "tail_index": alpha})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Tail Ratio
# ---------------------------------------------------------------------------

def tail_ratio(
    returns: ArrayLike,
    quantile: float = 0.05,
) -> float:
    """
    Tính tỷ lệ đuôi phải / đuôi trái.

    tail_ratio = |quantile(1 - q)| / |quantile(q)|

    > 1 → đuôi phải lớn hơn đuôi trái
    < 1 → đuôi trái lớn hơn (downside risk lớn hơn)

    Parameters
    ----------
    quantile : ngưỡng đuôi (default 0.05 → 5th / 95th percentile)
    """
    r = pd.Series(returns).dropna().values
    right = abs(np.quantile(r, 1 - quantile))
    left = abs(np.quantile(r, quantile))
    return float(right / left) if left != 0 else np.nan


# ---------------------------------------------------------------------------
# POT / Exceedance Threshold
# ---------------------------------------------------------------------------

def mean_excess_function(
    returns: ArrayLike,
    n_thresholds: int = 50,
) -> pd.DataFrame:
    """
    Tính Mean Excess Function (MEF) để chọn ngưỡng cho GPD/POT.

    MEF(u) = E[X - u | X > u]

    Vùng ngưỡng tốt: MEF tuyến tính và tăng dần.

    Parameters
    ----------
    returns      : chuỗi return (lấy losses = -return âm)
    n_thresholds : số điểm ngưỡng

    Returns
    -------
    pd.DataFrame với cột ['threshold', 'mean_excess', 'n_exceedances']
    """
    r = pd.Series(returns).dropna()
    losses = np.abs(r[r < 0].values)
    u_min, u_max = np.quantile(losses, 0.5), np.quantile(losses, 0.99)
    thresholds = np.linspace(u_min, u_max, n_thresholds)
    rows = []
    for u in thresholds:
        exceedances = losses[losses > u] - u
        if len(exceedances) >= 5:
            rows.append(
                {
                    "threshold": u,
                    "mean_excess": float(exceedances.mean()),
                    "n_exceedances": len(exceedances),
                }
            )
    return pd.DataFrame(rows)


def select_pot_threshold(
    returns: ArrayLike,
    method: str = "percentile",
    percentile: float = 0.90,
) -> float:
    """
    Chọn ngưỡng u cho mô hình POT (Peaks Over Threshold).

    Parameters
    ----------
    method      : 'percentile' → dùng phân vị của losses
                  'std'        → mean + k*std
    percentile  : phân vị nếu method='percentile' (default 0.90)

    Returns
    -------
    float – ngưỡng u
    """
    r = pd.Series(returns).dropna()
    losses = np.abs(r[r < 0].values)
    if method == "percentile":
        return float(np.quantile(losses, percentile))
    elif method == "std":
        return float(losses.mean() + 2 * losses.std())
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'percentile' or 'std'.")


def pot_exceedances(
    returns: ArrayLike,
    threshold: float,
) -> np.ndarray:
    """
    Trả về các giá trị vượt ngưỡng (exceedances) cho mô hình POT.

    Parameters
    ----------
    threshold : ngưỡng u

    Returns
    -------
    np.ndarray – (X - u) với X > u, X là losses (dương)
    """
    r = pd.Series(returns).dropna()
    losses = np.abs(r[r < 0].values)
    return losses[losses > threshold] - threshold


# ---------------------------------------------------------------------------
# Tail Probability & Quantile
# ---------------------------------------------------------------------------

def tail_probability(
    returns: ArrayLike,
    threshold: float,
    side: str = "left",
) -> float:
    """
    Tính xác suất thực nghiệm P(R ≤ threshold) hoặc P(R ≥ threshold).

    Parameters
    ----------
    threshold : mức ngưỡng
    side      : 'left' → P(R ≤ threshold), 'right' → P(R ≥ threshold)
    """
    r = pd.Series(returns).dropna().values
    if side == "left":
        return float(np.mean(r <= threshold))
    return float(np.mean(r >= threshold))


def tail_quantile(
    returns: ArrayLike,
    p: float,
) -> float:
    """
    Trả về quantile thực nghiệm tại mức p.

    Parameters
    ----------
    p : xác suất (0 < p < 1), ví dụ p=0.01 → 1st percentile
    """
    r = pd.Series(returns).dropna().values
    return float(np.quantile(r, p))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def tail_summary(
    returns: ArrayLike,
    confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99),
    hill_k_fraction: float = 0.10,
) -> pd.Series:
    """
    Tổng hợp các đặc trưng đuôi vào một Series.

    Parameters
    ----------
    returns           : chuỗi return
    confidence_levels : các mức tin cậy cần tính VaR/CVaR
    hill_k_fraction   : tỷ lệ đuôi dùng cho Hill estimator

    Returns
    -------
    pd.Series bao gồm VaR/CVaR ở nhiều mức, tail_index,
              tail_ratio, pot_threshold_90pct
    """
    r = pd.Series(returns).dropna()
    result = {}

    for cl in confidence_levels:
        pct = int(cl * 100)
        result[f"var_{pct}"] = empirical_var(r, confidence=cl)
        result[f"cvar_{pct}"] = empirical_cvar(r, confidence=cl)

    result["tail_index_hill"] = hill_estimator(r, k_fraction=hill_k_fraction)
    result["tail_ratio_5pct"] = tail_ratio(r, quantile=0.05)
    result["pot_threshold_90pct"] = select_pot_threshold(r, percentile=0.90)
    result["n_exceedances_90pct"] = len(
        pot_exceedances(r, result["pot_threshold_90pct"])
    )

    return pd.Series(result)