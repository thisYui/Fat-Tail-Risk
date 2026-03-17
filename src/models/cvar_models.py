"""
models/cvar_models.py
----------------------
Conditional VaR / Expected Shortfall (ES) theo các phương pháp:

  - Historical ES
  - Parametric ES (Normal, Student-t, NIG)
  - GARCH-ES
  - Filtered HS-ES
  - Super-quantile / CVaR decomposition
  - CVaRResult dataclass
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple
from dataclasses import dataclass

from .distribution import fit_normal, fit_student_t, fit_nig
from .var_models import (
    VaRResult,
    hs_var,
    parametric_var,
    garch_var,
    filtered_hs_var,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CVaRResult:
    """
    Kết quả ước lượng CVaR / Expected Shortfall.

    Attributes
    ----------
    cvar        : giá trị CVaR (âm = lỗ kỳ vọng)
    var         : VaR tương ứng
    confidence  : mức tin cậy
    method      : tên phương pháp
    n_tail_obs  : số quan sát trong đuôi (với phương pháp lịch sử)
    """
    cvar: float
    var: float
    confidence: float
    method: str
    n_tail_obs: Optional[int] = None
    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}

    @property
    def es_ratio(self) -> float:
        """Tỷ lệ CVaR / VaR – đo mức độ tail risk vượt quá VaR."""
        return self.cvar / self.var if self.var != 0 else np.nan

    def as_series(self) -> pd.Series:
        return pd.Series({
            "cvar": self.cvar,
            "var": self.var,
            "es_ratio": self.es_ratio,
            "confidence": self.confidence,
            "method": self.method,
            "n_tail_obs": self.n_tail_obs,
            **self.params,
        })


# ---------------------------------------------------------------------------
# 1. Historical CVaR
# ---------------------------------------------------------------------------

def historical_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
    window: Optional[int] = None,
) -> CVaRResult:
    """
    Historical Expected Shortfall: E[R | R ≤ VaR].

    Parameters
    ----------
    returns    : chuỗi return lịch sử
    confidence : mức tin cậy
    window     : số ngày gần nhất (None = toàn bộ)
    """
    r = pd.Series(returns).dropna().values
    if window is not None:
        r = r[-window:]

    var_res = hs_var(r, confidence)
    var = var_res.var
    tail = r[r <= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var

    return CVaRResult(
        cvar=cvar,
        var=var,
        confidence=confidence,
        method="historical",
        n_tail_obs=len(tail),
        params={"n_obs": len(r)},
    )


def rolling_historical_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.Series:
    """
    Rolling Historical CVaR trên toàn chuỗi.

    Returns
    -------
    pd.Series – CVaR tại mỗi điểm thời gian
    """
    alpha = 1 - confidence

    def _cvar(arr):
        q = np.quantile(arr, alpha)
        tail = arr[arr <= q]
        return float(tail.mean()) if len(tail) > 0 else q

    return returns.rolling(window).apply(_cvar, raw=True)


# ---------------------------------------------------------------------------
# 2. Parametric CVaR
# ---------------------------------------------------------------------------

def parametric_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
    dist: str = "normal",
) -> CVaRResult:
    """
    Parametric CVaR bằng công thức giải tích.

    Parameters
    ----------
    dist : 'normal' hoặc 'student_t'

    Notes
    -----
    Normal:    ES = μ - σ · φ(z_α) / α
    Student-t: ES = -loc + scale · (f_t(z_α) / α) · (df + z_α²) / (df - 1)
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    alpha = 1 - confidence

    if dist == "normal":
        fit = fit_normal(r)
        mu, sigma = fit.params["mu"], fit.params["sigma"]
        z_alpha = stats.norm.ppf(alpha)
        phi = stats.norm.pdf(z_alpha)
        cvar = float(mu - sigma * phi / alpha)
        var = float(stats.norm.ppf(alpha, loc=mu, scale=sigma))
        params = {"mu": mu, "sigma": sigma}

    elif dist == "student_t":
        fit = fit_student_t(r)
        df = fit.params["df"]
        loc = fit.params["loc"]
        scale = fit.params["scale"]

        # Quantile của standardised t
        t_alpha = stats.t.ppf(alpha, df=df)
        # PDF tại quantile
        f_t = stats.t.pdf(t_alpha, df=df)
        # Closed-form ES cho standardised t
        es_std = -(f_t / alpha) * (df + t_alpha**2) / (df - 1)
        cvar = float(loc + scale * es_std)
        var = float(stats.t.ppf(alpha, df=df, loc=loc, scale=scale))
        params = {"df": df, "loc": loc, "scale": scale}

    elif dist == "nig":
        fit = fit_nig(r)
        a = fit.params["alpha"]
        b = fit.params["beta"]
        loc = fit.params["loc"]
        scale = fit.params["scale"]
        # NIG không có closed-form ES → dùng numerical integration
        var = float(stats.norminvgauss.ppf(alpha, a, b, loc=loc, scale=scale))
        x_grid = np.linspace(r.min() * 2, var, 2000)
        pdf_vals = stats.norminvgauss.pdf(x_grid, a, b, loc=loc, scale=scale)
        cdf_at_var = float(stats.norminvgauss.cdf(var, a, b, loc=loc, scale=scale))
        cvar = float(np.trapz(x_grid * pdf_vals, x_grid) / cdf_at_var)
        params = {"alpha_nig": a, "beta_nig": b, "loc": loc, "scale": scale}

    else:
        raise ValueError(f"Unsupported dist: '{dist}'")

    return CVaRResult(
        cvar=cvar,
        var=var,
        confidence=confidence,
        method=f"parametric_{dist}",
        params=params,
    )


# ---------------------------------------------------------------------------
# 3. GARCH-CVaR
# ---------------------------------------------------------------------------

def garch_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
    omega: float = 1e-6,
    alpha_garch: float = 0.09,
    beta_garch: float = 0.90,
    dist: str = "normal",
) -> CVaRResult:
    """
    GARCH(1,1)-CVaR: ES điều kiện dựa trên phương sai GARCH.

    Dùng cùng phương sai một bước tiếp theo từ GARCH, sau đó
    tính ES theo phân phối lý thuyết.

    Parameters
    ----------
    dist : 'normal' hoặc 'student_t'
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    alpha_cl = 1 - confidence
    mu = float(r.mean())

    # GARCH filtering
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = omega / max(1 - alpha_garch - beta_garch, 1e-8)
    for t in range(1, n):
        eps2 = (r[t - 1] - mu) ** 2
        sigma2[t] = omega + alpha_garch * eps2 + beta_garch * sigma2[t - 1]

    eps_last = r[-1] - mu
    sigma2_next = omega + alpha_garch * eps_last**2 + beta_garch * sigma2[-1]
    sigma_next = float(np.sqrt(sigma2_next))

    if dist == "normal":
        z_alpha = stats.norm.ppf(alpha_cl)
        phi = stats.norm.pdf(z_alpha)
        cvar = float(mu - sigma_next * phi / alpha_cl)
        var = float(mu + sigma_next * z_alpha)

    elif dist == "student_t":
        fit = fit_student_t(r)
        df = fit.params["df"]
        t_alpha = stats.t.ppf(alpha_cl, df=df)
        f_t = stats.t.pdf(t_alpha, df=df)
        es_std = -(f_t / alpha_cl) * (df + t_alpha**2) / (df - 1)
        scale_factor = np.sqrt((df - 2) / df) if df > 2 else 1.0
        cvar = float(mu + sigma_next * es_std / scale_factor)
        var = float(mu + sigma_next * t_alpha / scale_factor)
    else:
        raise ValueError(f"Unsupported dist: '{dist}'")

    return CVaRResult(
        cvar=cvar,
        var=var,
        confidence=confidence,
        method=f"garch_{dist}",
        params={"sigma_next": sigma_next},
    )


# ---------------------------------------------------------------------------
# 4. Filtered HS-CVaR
# ---------------------------------------------------------------------------

def filtered_hs_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
    omega: float = 1e-6,
    alpha_garch: float = 0.09,
    beta_garch: float = 0.90,
) -> CVaRResult:
    """
    Filtered Historical Simulation CVaR.

    Cùng logic với FHS-VaR nhưng lấy trung bình đuôi của z.
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    n = len(r)
    mu = float(r.mean())
    alpha_cl = 1 - confidence

    sigma2 = np.empty(n)
    sigma2[0] = omega / max(1 - alpha_garch - beta_garch, 1e-8)
    for t in range(1, n):
        eps = r[t - 1] - mu
        sigma2[t] = omega + alpha_garch * eps**2 + beta_garch * sigma2[t - 1]

    sigma = np.sqrt(sigma2)
    z = (r - mu) / sigma

    eps_last = r[-1] - mu
    sigma2_next = omega + alpha_garch * eps_last**2 + beta_garch * sigma2[-1]
    sigma_next = float(np.sqrt(sigma2_next))

    z_quantile = float(np.quantile(z, alpha_cl))
    tail_z = z[z <= z_quantile]
    cvar_z = float(tail_z.mean()) if len(tail_z) > 0 else z_quantile
    var = float(mu + sigma_next * z_quantile)
    cvar = float(mu + sigma_next * cvar_z)

    return CVaRResult(
        cvar=cvar,
        var=var,
        confidence=confidence,
        method="filtered_hs",
        n_tail_obs=len(tail_z),
        params={"sigma_next": sigma_next, "cvar_z": cvar_z},
    )


# ---------------------------------------------------------------------------
# CVaR decomposition: contribution by sub-portfolio
# ---------------------------------------------------------------------------

def cvar_contribution(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    confidence: float = 0.95,
) -> pd.Series:
    """
    Tính Component CVaR – đóng góp của từng tài sản vào CVaR danh mục.

    Phương pháp: Euler allocation
    CVaR_i = E[R_i | R_p ≤ VaR_p]

    Parameters
    ----------
    weights        : trọng số danh mục (n_assets,)
    returns_matrix : ma trận return (n_obs × n_assets)

    Returns
    -------
    pd.Series – CVaR đóng góp của từng tài sản (tổng = CVaR danh mục)
    """
    w = np.asarray(weights) / np.sum(weights)
    R = np.asarray(returns_matrix)
    port_ret = R @ w
    alpha = 1 - confidence
    var = float(np.quantile(port_ret, alpha))
    tail_mask = port_ret <= var

    if tail_mask.sum() == 0:
        return pd.Series(np.zeros(len(w)))

    component_cvar = (R[tail_mask] * w).mean(axis=0)
    return pd.Series(component_cvar, name="component_cvar")


# ---------------------------------------------------------------------------
# Compare all CVaR methods
# ---------------------------------------------------------------------------

def compare_cvar_methods(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Tính CVaR bằng tất cả các phương pháp và so sánh.

    Returns
    -------
    pd.DataFrame với index = method, columns = [var, cvar, es_ratio]
    """
    methods = {
        "historical": lambda r: historical_cvar(r, confidence),
        "parametric_normal": lambda r: parametric_cvar(r, confidence, "normal"),
        "parametric_t": lambda r: parametric_cvar(r, confidence, "student_t"),
        "garch_normal": lambda r: garch_cvar(r, confidence, dist="normal"),
        "garch_t": lambda r: garch_cvar(r, confidence, dist="student_t"),
        "filtered_hs": lambda r: filtered_hs_cvar(r, confidence),
    }
    rows = []
    for name, fn in methods.items():
        try:
            res = fn(returns)
            rows.append({
                "method": name,
                "var": res.var,
                "cvar": res.cvar,
                "es_ratio": res.es_ratio,
            })
        except Exception as e:
            rows.append({"method": name, "var": np.nan, "cvar": np.nan, "error": str(e)})

    return pd.DataFrame(rows).set_index("method").sort_values("cvar")