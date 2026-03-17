"""
models/evt.py
--------------
Extreme Value Theory (EVT) cho phân tích tail risk:

  - Block Maxima / GEV (Generalized Extreme Value)
  - Peaks-Over-Threshold / GPD (Generalized Pareto Distribution)
  - EVT-VaR và EVT-CVaR
  - Return level (chu kỳ lặp lại)
  - Threshold selection diagnostics
  - EVTResult dataclass
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Optional, Tuple
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class GEVResult:
    """Kết quả fit Generalized Extreme Value (Block Maxima)."""
    xi: float       # shape (xi > 0 = Fréchet/fat-tail)
    mu: float       # location
    sigma: float    # scale
    log_lik: float
    aic: float
    bic: float
    n_blocks: int

    def return_level(self, T: float) -> float:
        """
        Return level tương ứng với chu kỳ T năm (hoặc T block).

        z_T = μ - σ/ξ · [1 - (-ln(1 - 1/T))^{-ξ}]
        """
        p = 1 - 1 / T
        if np.isclose(self.xi, 0):
            return float(self.mu - self.sigma * np.log(-np.log(p)))
        return float(
            self.mu - self.sigma / self.xi
            * (1 - (-np.log(p)) ** (-self.xi))
        )


@dataclass
class GPDResult:
    """Kết quả fit Generalized Pareto Distribution (POT)."""
    xi: float       # shape (tail index)
    sigma: float    # scale
    threshold: float
    n_exceedances: int
    n_total: int
    log_lik: float
    aic: float
    bic: float

    @property
    def exceedance_rate(self) -> float:
        """Tỷ lệ phần trăm vượt ngưỡng."""
        return self.n_exceedances / self.n_total

    def var(self, confidence: float = 0.95) -> float:
        """
        EVT-VaR từ GPD fit.

        VaR_p = u + σ/ξ · [(n/N_u · (1-p))^{-ξ} - 1]

        Parameters
        ----------
        confidence : mức tin cậy (p)
        """
        p = confidence
        Nu = self.n_exceedances
        n = self.n_total
        u = self.threshold
        if np.isclose(self.xi, 0):
            return float(u - self.sigma * np.log(Nu / n * (1 - p)))
        return float(
            u + self.sigma / self.xi
            * ((n / Nu * (1 - p)) ** (-self.xi) - 1)
        )

    def cvar(self, confidence: float = 0.95) -> float:
        """
        EVT-CVaR (Expected Shortfall) từ GPD fit.

        ES_p = VaR_p / (1 - ξ) + (σ - ξ·u) / (1 - ξ)
        """
        var_p = self.var(confidence)
        if self.xi >= 1:
            return np.inf
        return float(
            var_p / (1 - self.xi)
            + (self.sigma - self.xi * self.threshold) / (1 - self.xi)
        )

    def return_level(self, T_years: float, trading_days: int = 252) -> float:
        """
        Loss level tương ứng với chu kỳ T_years năm.

        Parameters
        ----------
        T_years      : chu kỳ (năm)
        trading_days : số ngày giao dịch / năm
        """
        # Xác suất vượt mức trong 1 ngày
        p_exceed_1day = 1 / (T_years * trading_days)
        confidence = 1 - p_exceed_1day
        return self.var(confidence)


# ---------------------------------------------------------------------------
# Block Maxima – GEV
# ---------------------------------------------------------------------------

def fit_gev(
    returns: np.ndarray,
    block_size: int = 21,
    side: str = "left",
) -> GEVResult:
    """
    Fit GEV lên block maxima của losses.

    Parameters
    ----------
    block_size : kích thước mỗi block (21 = monthly, 63 = quarterly)
    side       : 'left' = losses (âm → dương), 'right' = gains

    Returns
    -------
    GEVResult
    """
    r = np.asarray(returns)[~np.isnan(returns)]

    if side == "left":
        data = -r   # chuyển loss thành dương
    else:
        data = r

    # Chia thành blocks và lấy max
    n_blocks = len(data) // block_size
    blocks = data[: n_blocks * block_size].reshape(n_blocks, block_size)
    block_maxima = blocks.max(axis=1)

    # Fit GEV bằng scipy
    xi, loc, sigma = stats.genextreme.fit(block_maxima)
    ll = float(stats.genextreme.logpdf(block_maxima, xi, loc=loc, scale=sigma).sum())
    k = 3

    return GEVResult(
        xi=float(xi),
        mu=float(loc),
        sigma=float(sigma),
        log_lik=ll,
        aic=2 * k - 2 * ll,
        bic=k * np.log(n_blocks) - 2 * ll,
        n_blocks=n_blocks,
    )


# ---------------------------------------------------------------------------
# Peaks-Over-Threshold – GPD
# ---------------------------------------------------------------------------

def fit_gpd_pot(
    returns: np.ndarray,
    threshold: Optional[float] = None,
    threshold_quantile: float = 0.90,
    side: str = "left",
) -> GPDResult:
    """
    Fit GPD lên phần vượt ngưỡng (Peaks-Over-Threshold).

    Parameters
    ----------
    threshold          : ngưỡng u (None = tự động)
    threshold_quantile : phân vị losses để chọn ngưỡng
    side               : 'left' = loss tail, 'right' = gain tail

    Returns
    -------
    GPDResult
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    n = len(r)

    if side == "left":
        data = -r   # losses dương
    else:
        data = r

    if threshold is None:
        threshold = float(np.quantile(data, threshold_quantile))

    exceedances = data[data > threshold] - threshold
    n_exc = len(exceedances)

    if n_exc < 10:
        raise ValueError(
            f"Quá ít exceedances ({n_exc}) tại ngưỡng {threshold:.4f}. "
            "Hãy giảm threshold_quantile."
        )

    # Fit GPD (loc=0 cố định)
    xi, _, sigma = stats.genpareto.fit(exceedances, floc=0)
    ll = float(stats.genpareto.logpdf(exceedances, xi, loc=0, scale=sigma).sum())
    k = 2  # xi, sigma

    return GPDResult(
        xi=float(xi),
        sigma=float(sigma),
        threshold=float(threshold),
        n_exceedances=n_exc,
        n_total=n,
        log_lik=ll,
        aic=2 * k - 2 * ll,
        bic=k * np.log(n_exc) - 2 * ll,
    )


# ---------------------------------------------------------------------------
# EVT-VaR và EVT-CVaR wrapper
# ---------------------------------------------------------------------------

def evt_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    method: str = "pot",
    threshold_quantile: float = 0.90,
    block_size: int = 21,
) -> float:
    """
    EVT-VaR từ GPD (POT) hoặc GEV (Block Maxima).

    Parameters
    ----------
    method : 'pot' hoặc 'bm'
    """
    if method == "pot":
        gpd = fit_gpd_pot(returns, threshold_quantile=threshold_quantile)
        return gpd.var(confidence)
    elif method == "bm":
        gev = fit_gev(returns, block_size=block_size)
        # Chuyển confidence → return period T
        # P(max > z_T) = 1/T per block → rough approximation
        return gev.return_level(1 / (1 - confidence))
    else:
        raise ValueError(f"Unknown method: '{method}'")


def evt_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
    threshold_quantile: float = 0.90,
) -> float:
    """
    EVT-CVaR từ GPD fit.

    Returns
    -------
    float – Expected Shortfall (âm = lỗ)
    """
    gpd = fit_gpd_pot(returns, threshold_quantile=threshold_quantile)
    return gpd.cvar(confidence)


# ---------------------------------------------------------------------------
# Return levels
# ---------------------------------------------------------------------------

def return_levels(
    returns: np.ndarray,
    periods: Tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100),
    trading_days: int = 252,
    threshold_quantile: float = 0.90,
) -> pd.DataFrame:
    """
    Tính return level (mức lỗ tối đa) theo chu kỳ T năm.

    Parameters
    ----------
    periods       : các chu kỳ T năm cần tính
    trading_days  : số ngày giao dịch / năm

    Returns
    -------
    pd.DataFrame với cột ['period_years', 'return_level', 'var_equivalent']
    """
    gpd = fit_gpd_pot(returns, threshold_quantile=threshold_quantile)
    rows = []
    for T in periods:
        rl = gpd.return_level(T, trading_days=trading_days)
        # Chuyển sang loss (âm)
        rows.append({
            "period_years": T,
            "return_level_loss": -rl,
            "return_level_pct": -rl * 100,
        })
    return pd.DataFrame(rows).set_index("period_years")


# ---------------------------------------------------------------------------
# Threshold selection diagnostics
# ---------------------------------------------------------------------------

def threshold_stability_plot_data(
    returns: np.ndarray,
    n_thresholds: int = 40,
    min_exceedances: int = 20,
) -> pd.DataFrame:
    """
    Tính ổn định tham số GPD (xi, sigma*) theo ngưỡng u.

    Ngưỡng tốt: vùng xi và sigma* ổn định (không đổi theo u).

    sigma* = sigma - xi·u  (modified scale, ổn định nếu model đúng)

    Returns
    -------
    pd.DataFrame với cột ['threshold', 'xi', 'sigma', 'sigma_star',
                           'n_exceedances', 'xi_se', 'sigma_se']
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    losses = -r[r < 0]
    u_min = np.quantile(losses, 0.50)
    u_max = np.quantile(losses, 0.98)
    thresholds = np.linspace(u_min, u_max, n_thresholds)

    rows = []
    for u in thresholds:
        exc = losses[losses > u] - u
        if len(exc) < min_exceedances:
            continue
        try:
            xi, _, sigma = stats.genpareto.fit(exc, floc=0)
            sigma_star = sigma - xi * u
            rows.append({
                "threshold": float(u),
                "xi": float(xi),
                "sigma": float(sigma),
                "sigma_star": float(sigma_star),
                "n_exceedances": len(exc),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def mean_excess_plot_data(
    returns: np.ndarray,
    n_thresholds: int = 50,
) -> pd.DataFrame:
    """
    Mean Excess Function (MEF) để trực quan hoá chọn ngưỡng.

    MEF(u) = E[X - u | X > u]

    Nếu MEF tuyến tính tăng → GPD/Pareto phù hợp phía trên u.

    Returns
    -------
    pd.DataFrame với ['threshold', 'mean_excess', 'n_exceedances']
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    losses = -r[r < 0]
    u_arr = np.linspace(np.quantile(losses, 0.3), np.quantile(losses, 0.99), n_thresholds)

    rows = []
    for u in u_arr:
        exc = losses[losses > u] - u
        if len(exc) >= 5:
            rows.append({
                "threshold": float(u),
                "mean_excess": float(exc.mean()),
                "n_exceedances": len(exc),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full EVT summary
# ---------------------------------------------------------------------------

def evt_summary(
    returns: np.ndarray,
    confidence_levels: Tuple[float, ...] = (0.95, 0.99, 0.999),
    threshold_quantile: float = 0.90,
    periods: Tuple[float, ...] = (1, 5, 10, 25, 50),
    trading_days: int = 252,
) -> pd.Series:
    """
    Tổng hợp toàn bộ kết quả EVT vào một Series.

    Parameters
    ----------
    confidence_levels  : các mức tính VaR/CVaR
    threshold_quantile : ngưỡng POT
    periods            : các chu kỳ return level (năm)

    Returns
    -------
    pd.Series: GPD params, VaR/CVaR nhiều mức, return levels
    """
    gpd = fit_gpd_pot(returns, threshold_quantile=threshold_quantile)
    result = {
        "gpd_xi": gpd.xi,
        "gpd_sigma": gpd.sigma,
        "gpd_threshold": gpd.threshold,
        "gpd_n_exceedances": gpd.n_exceedances,
        "gpd_exceedance_rate": gpd.exceedance_rate,
        "gpd_aic": gpd.aic,
    }

    for cl in confidence_levels:
        pct = int(cl * 100)
        result[f"evt_var_{pct}"] = gpd.var(cl)
        result[f"evt_cvar_{pct}"] = gpd.cvar(cl)

    for T in periods:
        rl = gpd.return_level(T, trading_days=trading_days)
        result[f"return_level_{T}yr"] = -rl

    return pd.Series(result)