"""
models/var_models.py
---------------------
Value-at-Risk (VaR) theo các phương pháp khác nhau:

  - Historical Simulation (HS)
  - Parametric VaR (Normal, Student-t, NIG)
  - GARCH-VaR (conditional VaR với volatility dynamics)
  - Filtered Historical Simulation (FHS)
  - Cornish-Fisher expansion (modified VaR)
  - VaRResult dataclass
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from .distribution import fit_normal, fit_student_t, fit_nig, DistributionFit


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class VaRResult:
    """
    Kết quả ước lượng VaR.

    Attributes
    ----------
    var         : giá trị VaR (âm = lỗ)
    confidence  : mức tin cậy
    method      : tên phương pháp
    horizon     : số ngày tính VaR
    params      : tham số thêm của model
    """
    var: float
    confidence: float
    method: str
    horizon: int = 1
    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}

    def scale_to_horizon(self, h: int) -> "VaRResult":
        """Scale VaR từ 1 ngày lên h ngày bằng quy tắc sqrt(h)."""
        return VaRResult(
            var=self.var * np.sqrt(h),
            confidence=self.confidence,
            method=self.method + f"_scaled_{h}d",
            horizon=h,
            params=self.params,
        )

    def as_series(self) -> pd.Series:
        return pd.Series({
            "var": self.var,
            "confidence": self.confidence,
            "method": self.method,
            "horizon": self.horizon,
            **self.params,
        })


# ---------------------------------------------------------------------------
# 1. Historical Simulation (HS-VaR)
# ---------------------------------------------------------------------------

def hs_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    window: Optional[int] = None,
) -> VaRResult:
    """
    Historical Simulation VaR – phân vị thực nghiệm.

    Parameters
    ----------
    returns    : chuỗi return lịch sử
    confidence : mức tin cậy (0.95 = VaR 95%)
    window     : số ngày gần nhất dùng để tính (None = toàn bộ)

    Returns
    -------
    VaRResult – var là giá trị âm (lỗ)
    """
    r = pd.Series(returns).dropna().values
    if window is not None:
        r = r[-window:]
    alpha = 1 - confidence
    var = float(np.quantile(r, alpha))
    return VaRResult(
        var=var,
        confidence=confidence,
        method="historical_simulation",
        params={"n_obs": len(r), "window": window},
    )


def rolling_hs_var(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.Series:
    """
    Rolling Historical Simulation VaR trên toàn chuỗi.

    Parameters
    ----------
    returns    : chuỗi return có DatetimeIndex
    window     : cửa sổ rolling (ngày)

    Returns
    -------
    pd.Series – VaR tại mỗi điểm thời gian
    """
    alpha = 1 - confidence
    return returns.rolling(window).quantile(alpha)


# ---------------------------------------------------------------------------
# 2. Parametric VaR
# ---------------------------------------------------------------------------

def parametric_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    dist: str = "normal",
) -> VaRResult:
    """
    Parametric VaR bằng cách fit phân phối và lấy quantile.

    Parameters
    ----------
    dist : 'normal', 'student_t', 'nig'

    Returns
    -------
    VaRResult
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    alpha = 1 - confidence

    if dist == "normal":
        fit = fit_normal(r)
        mu, sigma = fit.params["mu"], fit.params["sigma"]
        var = float(stats.norm.ppf(alpha, loc=mu, scale=sigma))
        params = {"mu": mu, "sigma": sigma}

    elif dist == "student_t":
        fit = fit_student_t(r)
        df = fit.params["df"]
        loc = fit.params["loc"]
        scale = fit.params["scale"]
        var = float(stats.t.ppf(alpha, df=df, loc=loc, scale=scale))
        params = {"df": df, "loc": loc, "scale": scale}

    elif dist == "nig":
        fit = fit_nig(r)
        a = fit.params["alpha"]
        b = fit.params["beta"]
        loc = fit.params["loc"]
        scale = fit.params["scale"]
        var = float(stats.norminvgauss.ppf(alpha, a, b, loc=loc, scale=scale))
        params = {"alpha": a, "beta": b, "loc": loc, "scale": scale}

    else:
        raise ValueError(f"Unknown distribution: '{dist}'")

    return VaRResult(
        var=var,
        confidence=confidence,
        method=f"parametric_{dist}",
        params=params,
    )


# ---------------------------------------------------------------------------
# 3. Cornish-Fisher VaR (Modified VaR)
# ---------------------------------------------------------------------------

def cornish_fisher_var(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> VaRResult:
    """
    Cornish-Fisher expansion (Modified VaR).

    Điều chỉnh quantile Normal bằng skewness và kurtosis thực nghiệm:

    z_CF = z + (z²-1)/6·S + (z³-3z)/24·K - (2z³-5z)/36·S²

    Trong đó S = skewness, K = excess kurtosis, z = Normal quantile.

    Tốt hơn Normal VaR khi phân phối bất đối xứng hoặc fat-tail.
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)

    S = float(stats.skew(r))
    K = float(stats.kurtosis(r, fisher=True))  # excess kurtosis

    z_cf = (
        z
        + (z**2 - 1) / 6 * S
        + (z**3 - 3 * z) / 24 * K
        - (2 * z**3 - 5 * z) / 36 * S**2
    )

    mu = float(r.mean())
    sigma = float(r.std())
    var = float(mu + sigma * z_cf)

    return VaRResult(
        var=var,
        confidence=confidence,
        method="cornish_fisher",
        params={"skewness": S, "excess_kurtosis": K, "z_cf": z_cf},
    )


# ---------------------------------------------------------------------------
# 4. GARCH-VaR
# ---------------------------------------------------------------------------

def garch_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    omega: float = 1e-6,
    alpha_garch: float = 0.09,
    beta_garch: float = 0.90,
    dist: str = "normal",
    horizon: int = 1,
) -> VaRResult:
    """
    GARCH(1,1)-VaR: dự báo VaR dựa trên phương sai điều kiện.

    σ²_{T+1} = ω + α·ε²_T + β·σ²_T  (one-step ahead)

    Parameters
    ----------
    omega       : GARCH ω
    alpha_garch : GARCH α (ARCH term)
    beta_garch  : GARCH β (GARCH term)
    dist        : phân phối innovation ('normal' hoặc 'student_t')
    horizon     : số ngày tính VaR (1 = next-day)

    Returns
    -------
    VaRResult với VaR 1-ngày tới
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    alpha_cl = 1 - confidence

    # Ước lượng chuỗi phương sai điều kiện
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = omega / max(1 - alpha_garch - beta_garch, 1e-8)

    for t in range(1, n):
        eps2 = (r[t - 1] - r.mean()) ** 2
        sigma2[t] = omega + alpha_garch * eps2 + beta_garch * sigma2[t - 1]

    # One-step-ahead forecast
    eps2_last = (r[-1] - r.mean()) ** 2
    sigma2_next = omega + alpha_garch * eps2_last + beta_garch * sigma2[-1]
    sigma_next = float(np.sqrt(sigma2_next))

    # Scale sang h-step bằng sqrt(h)
    sigma_h = sigma_next * np.sqrt(horizon)

    mu = float(r.mean())

    if dist == "normal":
        var = float(mu * horizon + sigma_h * stats.norm.ppf(alpha_cl))
    elif dist == "student_t":
        fit = fit_student_t(r)
        df = fit.params["df"]
        # Standardised quantile
        t_q = float(stats.t.ppf(alpha_cl, df=df))
        # Scale t-quantile
        scale_factor = np.sqrt((df - 2) / df) if df > 2 else 1.0
        var = float(mu * horizon + sigma_h * t_q / scale_factor)
    else:
        raise ValueError(f"Unsupported dist: '{dist}'")

    return VaRResult(
        var=var,
        confidence=confidence,
        method=f"garch_{dist}",
        horizon=horizon,
        params={
            "sigma2_next": sigma2_next,
            "omega": omega,
            "alpha": alpha_garch,
            "beta": beta_garch,
        },
    )


# ---------------------------------------------------------------------------
# 5. Filtered Historical Simulation (FHS)
# ---------------------------------------------------------------------------

def filtered_hs_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    omega: float = 1e-6,
    alpha_garch: float = 0.09,
    beta_garch: float = 0.90,
) -> VaRResult:
    """
    Filtered Historical Simulation (Barone-Adesi et al. 1999).

    Bước 1: Fit GARCH → chuỗi standardised residuals z_t = ε_t / σ_t
    Bước 2: Bootstrap từ {z_t}
    Bước 3: VaR = μ + σ_{T+1} × quantile(z_t)

    Kết hợp linh hoạt của lịch sử (FHS) và động lực học vol (GARCH).
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    n = len(r)
    mu = float(r.mean())
    alpha_cl = 1 - confidence

    # GARCH filtering
    sigma2 = np.empty(n)
    sigma2[0] = omega / max(1 - alpha_garch - beta_garch, 1e-8)
    for t in range(1, n):
        eps = r[t - 1] - mu
        sigma2[t] = omega + alpha_garch * eps**2 + beta_garch * sigma2[t - 1]

    sigma = np.sqrt(sigma2)
    z = (r - mu) / sigma   # standardised residuals

    # One-step-ahead vol
    eps_last = r[-1] - mu
    sigma2_next = omega + alpha_garch * eps_last**2 + beta_garch * sigma2[-1]
    sigma_next = float(np.sqrt(sigma2_next))

    # VaR từ empirical quantile của z
    z_quantile = float(np.quantile(z, alpha_cl))
    var = float(mu + sigma_next * z_quantile)

    return VaRResult(
        var=var,
        confidence=confidence,
        method="filtered_hs",
        params={
            "sigma_next": sigma_next,
            "z_quantile": z_quantile,
            "n_residuals": len(z),
        },
    )


# ---------------------------------------------------------------------------
# Compare all VaR methods
# ---------------------------------------------------------------------------

def compare_var_methods(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Tính VaR bằng tất cả các phương pháp và so sánh.

    Returns
    -------
    pd.DataFrame với index = method, columns = var + params
    """
    methods = {
        "hs": lambda r: hs_var(r, confidence),
        "parametric_normal": lambda r: parametric_var(r, confidence, "normal"),
        "parametric_t": lambda r: parametric_var(r, confidence, "student_t"),
        "cornish_fisher": lambda r: cornish_fisher_var(r, confidence),
        "garch_normal": lambda r: garch_var(r, confidence, dist="normal"),
        "garch_t": lambda r: garch_var(r, confidence, dist="student_t"),
        "filtered_hs": lambda r: filtered_hs_var(r, confidence),
    }
    rows = []
    for name, fn in methods.items():
        try:
            res = fn(returns)
            rows.append({"method": name, "var": res.var})
        except Exception as e:
            rows.append({"method": name, "var": np.nan, "error": str(e)})

    df = pd.DataFrame(rows).set_index("method")
    return df.sort_values("var")