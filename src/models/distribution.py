"""
models/distribution.py
-----------------------
Fitting và so sánh các phân phối thống kê lên chuỗi return:

  - fit_normal        : MLE cho Normal
  - fit_student_t     : MLE cho Student-t
  - fit_skewed_t      : MLE cho Skewed Student-t (Hansen)
  - fit_nig           : MLE cho Normal Inverse Gaussian
  - fit_gpd           : MLE cho Generalized Pareto (tail only)
  - fit_all           : fit nhiều phân phối và xếp hạng theo AIC/BIC
  - DistributionFit   : dataclass kết quả fit
  - GoodnessOfFit     : KS-test, AD-test, AIC, BIC
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DistributionFit:
    """
    Kết quả fitting một phân phối lên dữ liệu.

    Attributes
    ----------
    dist_name  : tên phân phối
    params     : dict tham số ước lượng
    log_lik    : log-likelihood
    aic        : Akaike Information Criterion (thấp hơn = tốt hơn)
    bic        : Bayesian Information Criterion
    ks_stat    : Kolmogorov-Smirnov statistic
    ks_pvalue  : KS p-value (cao hơn = không bác bỏ phân phối)
    n_params   : số tham số
    n_obs      : số quan sát
    """
    dist_name: str
    params: Dict[str, float]
    log_lik: float
    aic: float
    bic: float
    ks_stat: float
    ks_pvalue: float
    n_params: int
    n_obs: int

    def summary(self) -> pd.Series:
        return pd.Series({
            "dist": self.dist_name,
            **self.params,
            "log_lik": self.log_lik,
            "aic": self.aic,
            "bic": self.bic,
            "ks_stat": self.ks_stat,
            "ks_pvalue": self.ks_pvalue,
        })


def _aic(log_lik: float, k: int) -> float:
    return 2 * k - 2 * log_lik


def _bic(log_lik: float, k: int, n: int) -> float:
    return k * np.log(n) - 2 * log_lik


def _ks(data: np.ndarray, cdf_fn) -> Tuple[float, float]:
    stat, pval = stats.kstest(data, cdf_fn)
    return float(stat), float(pval)


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------

def fit_normal(returns: np.ndarray) -> DistributionFit:
    """Fit phân phối Normal N(μ, σ²) bằng MLE (= sample mean/std)."""
    r = np.asarray(returns)[~np.isnan(returns)]
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    ll = float(stats.norm.logpdf(r, loc=mu, scale=sigma).sum())
    ks_s, ks_p = _ks(r, lambda x: stats.norm.cdf(x, loc=mu, scale=sigma))
    k = 2
    return DistributionFit(
        dist_name="normal",
        params={"mu": mu, "sigma": sigma},
        log_lik=ll,
        aic=_aic(ll, k),
        bic=_bic(ll, k, len(r)),
        ks_stat=ks_s, ks_pvalue=ks_p,
        n_params=k, n_obs=len(r),
    )


# ---------------------------------------------------------------------------
# Student-t
# ---------------------------------------------------------------------------

def fit_student_t(returns: np.ndarray) -> DistributionFit:
    """
    Fit Student-t(df, loc, scale) bằng MLE dùng scipy.

    Cho phép df tự do → tự động ước lượng độ dày đuôi.
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    df, loc, scale = stats.t.fit(r)
    ll = float(stats.t.logpdf(r, df=df, loc=loc, scale=scale).sum())
    ks_s, ks_p = _ks(r, lambda x: stats.t.cdf(x, df=df, loc=loc, scale=scale))
    k = 3
    return DistributionFit(
        dist_name="student_t",
        params={"df": float(df), "loc": float(loc), "scale": float(scale)},
        log_lik=ll,
        aic=_aic(ll, k),
        bic=_bic(ll, k, len(r)),
        ks_stat=ks_s, ks_pvalue=ks_p,
        n_params=k, n_obs=len(r),
    )


# ---------------------------------------------------------------------------
# Skewed Student-t (Hansen 1994) – numerical MLE
# ---------------------------------------------------------------------------

def _skewed_t_logpdf(
    x: np.ndarray,
    df: float,
    skew: float,
    loc: float,
    scale: float,
) -> np.ndarray:
    """Log-PDF của Skewed Student-t (Fernandez-Steel two-piece scale)."""
    z = (x - loc) / scale
    gamma = np.exp(skew)
    cut = gamma / (gamma + 1.0 / gamma)
    log_pdf = np.where(
        z < 0,
        stats.t.logpdf(z * gamma, df=df),
        stats.t.logpdf(z / gamma, df=df),
    ) + np.log(2) + np.log(gamma) - np.log(gamma + 1.0 / gamma) - np.log(scale)
    return log_pdf


def fit_skewed_t(returns: np.ndarray) -> DistributionFit:
    """Fit Skewed Student-t bằng numerical MLE."""
    r = np.asarray(returns)[~np.isnan(returns)]

    def neg_ll(params):
        df, skew, loc, scale = params
        if df <= 2 or scale <= 0:
            return 1e10
        return -float(_skewed_t_logpdf(r, df, skew, loc, scale).sum())

    # Khởi tạo từ Student-t fit
    df0, loc0, scale0 = stats.t.fit(r)
    x0 = [max(df0, 2.5), 0.0, loc0, scale0]
    bounds = [(2.01, 100), (-2, 2), (-1, 1), (1e-6, None)]
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)

    df, skew, loc, scale = res.x
    ll = -float(res.fun)
    ks_s, ks_p = _ks(
        r,
        lambda x: stats.norm.cdf(x, loc=loc, scale=scale)  # approximate KS
    )
    k = 4
    return DistributionFit(
        dist_name="skewed_t",
        params={"df": float(df), "skew": float(skew), "loc": float(loc), "scale": float(scale)},
        log_lik=ll,
        aic=_aic(ll, k),
        bic=_bic(ll, k, len(r)),
        ks_stat=ks_s, ks_pvalue=ks_p,
        n_params=k, n_obs=len(r),
    )


# ---------------------------------------------------------------------------
# Normal Inverse Gaussian (NIG)
# ---------------------------------------------------------------------------

def fit_nig(returns: np.ndarray) -> DistributionFit:
    """
    Fit Normal Inverse Gaussian (NIG) bằng MLE.

    NIG là phân phối linh hoạt với 4 tham số: α, β, μ, δ.
    Nắm bắt được cả fat-tail và skewness.
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    # scipy norminvgauss: a (alpha), b (beta), loc, scale
    a, b, loc, scale = stats.norminvgauss.fit(r)
    ll = float(stats.norminvgauss.logpdf(r, a, b, loc=loc, scale=scale).sum())
    ks_s, ks_p = _ks(
        r,
        lambda x: stats.norminvgauss.cdf(x, a, b, loc=loc, scale=scale),
    )
    k = 4
    return DistributionFit(
        dist_name="nig",
        params={"alpha": float(a), "beta": float(b), "loc": float(loc), "scale": float(scale)},
        log_lik=ll,
        aic=_aic(ll, k),
        bic=_bic(ll, k, len(r)),
        ks_stat=ks_s, ks_pvalue=ks_p,
        n_params=k, n_obs=len(r),
    )


# ---------------------------------------------------------------------------
# Generalized Pareto Distribution (GPD) – tail fitting
# ---------------------------------------------------------------------------

def fit_gpd(
    returns: np.ndarray,
    threshold: Optional[float] = None,
    threshold_quantile: float = 0.90,
) -> DistributionFit:
    """
    Fit GPD lên phần đuôi vượt ngưỡng (POT method).

    Parameters
    ----------
    threshold          : ngưỡng u (None → tự động từ quantile)
    threshold_quantile : phân vị để chọn ngưỡng tự động (default 0.90)

    Notes
    -----
    Chỉ fit trên losses (giá trị tuyệt đối của return âm).
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    losses = np.abs(r[r < 0])

    if threshold is None:
        threshold = float(np.quantile(losses, threshold_quantile))

    exceedances = losses[losses > threshold] - threshold
    if len(exceedances) < 10:
        raise ValueError(
            f"Too few exceedances ({len(exceedances)}) above threshold {threshold:.4f}. "
            "Try lowering threshold_quantile."
        )

    xi, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    ll = float(stats.genpareto.logpdf(exceedances, xi, loc=loc, scale=scale).sum())
    ks_s, ks_p = _ks(
        exceedances,
        lambda x: stats.genpareto.cdf(x, xi, loc=loc, scale=scale),
    )
    k = 2  # xi, scale (loc fixed = 0)
    return DistributionFit(
        dist_name="gpd",
        params={
            "xi": float(xi),
            "scale": float(scale),
            "threshold": float(threshold),
            "n_exceedances": len(exceedances),
        },
        log_lik=ll,
        aic=_aic(ll, k),
        bic=_bic(ll, k, len(exceedances)),
        ks_stat=ks_s, ks_pvalue=ks_p,
        n_params=k, n_obs=len(exceedances),
    )


# ---------------------------------------------------------------------------
# Fit all & rank
# ---------------------------------------------------------------------------

FITTERS = {
    "normal": fit_normal,
    "student_t": fit_student_t,
    "skewed_t": fit_skewed_t,
    "nig": fit_nig,
}


def fit_all(
    returns: np.ndarray,
    distributions: Optional[List[str]] = None,
    rank_by: str = "aic",
) -> pd.DataFrame:
    """
    Fit nhiều phân phối lên dữ liệu và xếp hạng theo AIC/BIC.

    Parameters
    ----------
    returns       : chuỗi return
    distributions : danh sách phân phối cần fit (None = tất cả)
    rank_by       : 'aic' hoặc 'bic'

    Returns
    -------
    pd.DataFrame đã sắp xếp tăng dần theo rank_by
    """
    if distributions is None:
        distributions = list(FITTERS.keys())

    rows = []
    for name in distributions:
        if name not in FITTERS:
            continue
        try:
            fit = FITTERS[name](returns)
            rows.append(fit.summary())
        except Exception as e:
            rows.append(pd.Series({"dist": name, "error": str(e)}))

    df = pd.DataFrame(rows).set_index("dist")
    if rank_by in df.columns:
        df = df.sort_values(rank_by)
    return df


def best_fit(
    returns: np.ndarray,
    rank_by: str = "aic",
) -> DistributionFit:
    """
    Trả về phân phối fit tốt nhất (theo AIC hoặc BIC).

    Parameters
    ----------
    rank_by : 'aic' hoặc 'bic'
    """
    best_name = None
    best_score = np.inf
    best_result = None

    for name, fitter in FITTERS.items():
        try:
            fit = fitter(returns)
            score = fit.aic if rank_by == "aic" else fit.bic
            if score < best_score:
                best_score = score
                best_name = name
                best_result = fit
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("All distribution fits failed.")
    return best_result


# ---------------------------------------------------------------------------
# Quantile & PDF helpers
# ---------------------------------------------------------------------------

def fitted_quantile(fit: DistributionFit, p: float) -> float:
    """
    Tính quantile tại mức p từ phân phối đã fit.

    Parameters
    ----------
    fit : DistributionFit object
    p   : xác suất (0 < p < 1)

    Returns
    -------
    float – quantile
    """
    p_map = fit.params
    if fit.dist_name == "normal":
        return float(stats.norm.ppf(p, loc=p_map["mu"], scale=p_map["sigma"]))
    elif fit.dist_name == "student_t":
        return float(stats.t.ppf(p, df=p_map["df"], loc=p_map["loc"], scale=p_map["scale"]))
    elif fit.dist_name == "nig":
        return float(stats.norminvgauss.ppf(
            p, p_map["alpha"], p_map["beta"], loc=p_map["loc"], scale=p_map["scale"]
        ))
    else:
        raise NotImplementedError(f"Quantile not implemented for '{fit.dist_name}'.")


def fitted_pdf(fit: DistributionFit, x: np.ndarray) -> np.ndarray:
    """
    Tính PDF tại các điểm x từ phân phối đã fit.

    Parameters
    ----------
    fit : DistributionFit
    x   : array các điểm cần tính PDF

    Returns
    -------
    np.ndarray – giá trị PDF
    """
    p = fit.params
    if fit.dist_name == "normal":
        return stats.norm.pdf(x, loc=p["mu"], scale=p["sigma"])
    elif fit.dist_name == "student_t":
        return stats.t.pdf(x, df=p["df"], loc=p["loc"], scale=p["scale"])
    elif fit.dist_name == "skewed_t":
        return np.exp(_skewed_t_logpdf(x, p["df"], p["skew"], p["loc"], p["scale"]))
    elif fit.dist_name == "nig":
        return stats.norminvgauss.pdf(x, p["alpha"], p["beta"], loc=p["loc"], scale=p["scale"])
    else:
        raise NotImplementedError(f"PDF not implemented for '{fit.dist_name}'.")