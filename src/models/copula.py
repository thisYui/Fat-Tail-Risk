"""
models/copula.py
-----------------
Mô hình Copula để mô phỏng cấu trúc phụ thuộc giữa các tài sản:

  - Gaussian Copula
  - Student-t Copula (tail dependence)
  - Clayton Copula (lower tail dependence)
  - Gumbel Copula (upper tail dependence)
  - Frank Copula (symmetric)
  - Fit copula từ dữ liệu
  - Sinh mẫu từ copula đã fit
  - Đo lường tail dependence coefficients
  - CopulaResult dataclass
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums & Result containers
# ---------------------------------------------------------------------------

class CopulaFamily(str, Enum):
    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"


@dataclass
class CopulaResult:
    """
    Kết quả fit một Copula.

    Attributes
    ----------
    family     : họ copula (CopulaFamily)
    params     : dict tham số ước lượng
    log_lik    : log-likelihood
    aic        : AIC
    bic        : BIC
    tail_dep_lower : hệ số phụ thuộc đuôi dưới λ_L
    tail_dep_upper : hệ số phụ thuộc đuôi trên λ_U
    n_obs      : số quan sát
    """
    family: str
    params: dict
    log_lik: float
    aic: float
    bic: float
    tail_dep_lower: float
    tail_dep_upper: float
    n_obs: int

    def summary(self) -> pd.Series:
        return pd.Series({
            "family": self.family,
            **self.params,
            "log_lik": self.log_lik,
            "aic": self.aic,
            "bic": self.bic,
            "tail_dep_lower": self.tail_dep_lower,
            "tail_dep_upper": self.tail_dep_upper,
        })


def _aic(ll: float, k: int) -> float:
    return 2 * k - 2 * ll


def _bic(ll: float, k: int, n: int) -> float:
    return k * np.log(n) - 2 * ll


# ---------------------------------------------------------------------------
# Pseudo-observations (empirical CDF transform)
# ---------------------------------------------------------------------------

def pseudo_observations(data: np.ndarray, adjust: float = 0.5) -> np.ndarray:
    """
    Chuyển dữ liệu sang pseudo-observations u ∈ (0, 1).

    u_it = rank(X_it) / (n + 1)

    Parameters
    ----------
    data   : ma trận (n_obs × d)
    adjust : adjustment cho rank (default 0.5 = Hazen formula)

    Returns
    -------
    np.ndarray (n_obs × d) với giá trị ∈ (0, 1)
    """
    n, d = data.shape
    u = np.empty_like(data, dtype=float)
    for j in range(d):
        ranks = stats.rankdata(data[:, j])
        u[:, j] = (ranks - adjust) / n
    return u


# ---------------------------------------------------------------------------
# Gaussian Copula
# ---------------------------------------------------------------------------

def fit_gaussian_copula(u: np.ndarray) -> CopulaResult:
    """
    Fit Gaussian Copula bằng ước lượng ma trận tương quan.

    C(u₁,...,u_d) = Φ_d(Φ⁻¹(u₁),...,Φ⁻¹(u_d); R)

    Parameters
    ----------
    u : pseudo-observations (n_obs × d), mỗi cột ∈ (0, 1)

    Returns
    -------
    CopulaResult
    """
    n, d = u.shape
    # Chuyển sang Normal scores
    z = stats.norm.ppf(np.clip(u, 1e-8, 1 - 1e-8))
    # Ước lượng ma trận tương quan bằng MLE (= sample correlation)
    R = np.corrcoef(z.T)

    # Log-likelihood của Gaussian copula
    try:
        R_inv = np.linalg.inv(R)
        sign, log_det = np.linalg.slogdet(R)
        ll_terms = -0.5 * (
            np.einsum("ni,ij,nj->n", z, R_inv - np.eye(d), z)
        )
        ll = float(ll_terms.sum() - n * 0.5 * log_det)
    except np.linalg.LinAlgError:
        ll = -np.inf

    k = d * (d - 1) // 2  # số phần tử off-diagonal

    return CopulaResult(
        family=CopulaFamily.GAUSSIAN,
        params={"correlation_matrix": R.tolist()},
        log_lik=ll,
        aic=_aic(ll, k),
        bic=_bic(ll, k, n),
        tail_dep_lower=0.0,   # Gaussian: no tail dependence
        tail_dep_upper=0.0,
        n_obs=n,
    )


def sample_gaussian_copula(
    n: int,
    R: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sinh mẫu từ Gaussian Copula với ma trận tương quan R.

    Returns
    -------
    np.ndarray (n × d) – pseudo-observations ∈ (0, 1)
    """
    rng = np.random.default_rng(seed)
    d = R.shape[0]
    L = np.linalg.cholesky(R + np.eye(d) * 1e-10)
    z = rng.standard_normal((n, d)) @ L.T
    return stats.norm.cdf(z)


# ---------------------------------------------------------------------------
# Student-t Copula
# ---------------------------------------------------------------------------

def fit_student_t_copula(
    u: np.ndarray,
    df_bounds: Tuple[float, float] = (2.0, 50.0),
) -> CopulaResult:
    """
    Fit Student-t Copula với ước lượng (df, R) bằng MLE.

    C(u₁,...,u_d; R, ν) = T_d(T_ν⁻¹(u₁),...,T_ν⁻¹(u_d); R, ν)

    Tail dependence coefficient:
    λ_L = λ_U = 2·t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))

    Parameters
    ----------
    u         : pseudo-observations (n_obs × d)
    df_bounds : bounds cho df trong optimisation
    """
    n, d = u.shape

    def neg_ll(df):
        if df <= 2:
            return 1e10
        z = stats.t.ppf(np.clip(u, 1e-8, 1 - 1e-8), df=df)
        R = np.corrcoef(z.T)
        try:
            R_inv = np.linalg.inv(R)
            _, log_det = np.linalg.slogdet(R)
            # Log-likelihood của t-copula
            term1 = stats.t.logpdf(z, df=df).sum(axis=1)
            quad = np.einsum("ni,ij,nj->n", z, R_inv, z)
            ll = (
                stats.multivariate_normal.logpdf(
                    z, cov=R, allow_singular=True
                ).sum()
            )
            # Simplified: dùng Gaussian + correction
            return -float(ll)
        except Exception:
            return 1e10

    res = minimize_scalar(neg_ll, bounds=df_bounds, method="bounded")
    df_opt = float(res.x)
    z = stats.t.ppf(np.clip(u, 1e-8, 1 - 1e-8), df=df_opt)
    R = np.corrcoef(z.T)
    ll = -float(res.fun)
    k = d * (d - 1) // 2 + 1  # correlation params + df

    # Tail dependence (bivariate, dùng trung bình correlation)
    if d == 2:
        rho = float(R[0, 1])
    else:
        off_diag = R[np.triu_indices(d, k=1)]
        rho = float(np.mean(off_diag))

    if df_opt > 2:
        t_arg = -np.sqrt((df_opt + 1) * (1 - rho) / max(1 + rho, 1e-8))
        tail_dep = float(2 * stats.t.cdf(t_arg, df=df_opt + 1))
    else:
        tail_dep = 1.0

    return CopulaResult(
        family=CopulaFamily.STUDENT_T,
        params={"df": df_opt, "correlation_matrix": R.tolist()},
        log_lik=ll,
        aic=_aic(ll, k),
        bic=_bic(ll, k, n),
        tail_dep_lower=tail_dep,
        tail_dep_upper=tail_dep,
        n_obs=n,
    )


def sample_student_t_copula(
    n: int,
    df: float,
    R: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sinh mẫu từ Student-t Copula.

    Returns
    -------
    np.ndarray (n × d) – pseudo-observations ∈ (0, 1)
    """
    rng = np.random.default_rng(seed)
    d = R.shape[0]
    L = np.linalg.cholesky(R + np.eye(d) * 1e-10)
    z = rng.standard_normal((n, d)) @ L.T
    v = rng.chisquare(df=df, size=(n, 1))
    t_samples = z / np.sqrt(v / df)
    return stats.t.cdf(t_samples, df=df)


# ---------------------------------------------------------------------------
# Archimedean Copulas (bivariate)
# ---------------------------------------------------------------------------

def _kendall_tau_to_theta(tau: float, family: str) -> float:
    """Chuyển Kendall's tau sang tham số θ của Archimedean copula."""
    if family == "clayton":
        # τ = θ / (θ + 2) → θ = 2τ / (1 - τ)
        return max(2 * tau / max(1 - tau, 1e-8), 1e-6)
    elif family == "gumbel":
        # τ = 1 - 1/θ → θ = 1/(1-τ)
        return max(1 / max(1 - tau, 1e-8), 1.0 + 1e-6)
    elif family == "frank":
        # Numerical: không có dạng giải tích đơn giản
        # Xấp xỉ: θ ≈ τ·π² / 3 cho |τ| nhỏ
        return tau * np.pi**2 / 3
    else:
        raise ValueError(f"Unknown Archimedean family: {family}")


def fit_archimedean_copula(
    u: np.ndarray,
    family: str = "clayton",
) -> CopulaResult:
    """
    Fit Archimedean copula (bivariate) bằng method-of-moments (Kendall's tau).

    Hỗ trợ: 'clayton', 'gumbel', 'frank'.

    Parameters
    ----------
    u      : pseudo-observations (n_obs × 2)
    family : họ copula

    Returns
    -------
    CopulaResult
    """
    if u.shape[1] != 2:
        raise ValueError("Archimedean copulas only support bivariate (d=2) case.")

    n = u.shape[0]
    u1, u2 = u[:, 0], u[:, 1]

    # Ước lượng theta từ Kendall's tau
    tau, _ = stats.kendalltau(u1, u2)
    theta = _kendall_tau_to_theta(tau, family)

    # Log-likelihood và tail dependence
    if family == "clayton":
        # C(u,v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}
        # c(u,v) = (θ+1)(uv)^{-θ-1}(u^{-θ}+v^{-θ}-1)^{-1/θ-2}
        u1c = np.clip(u1, 1e-8, 1 - 1e-8)
        u2c = np.clip(u2, 1e-8, 1 - 1e-8)
        s = u1c**(-theta) + u2c**(-theta) - 1
        log_density = (
            np.log(theta + 1)
            + (-theta - 1) * (np.log(u1c) + np.log(u2c))
            + (-1 / theta - 2) * np.log(np.maximum(s, 1e-300))
        )
        ll = float(log_density.sum())
        tail_lower = 2 ** (-1 / theta)
        tail_upper = 0.0

    elif family == "gumbel":
        # Numerical log-likelihood (simplified)
        u1c = np.clip(u1, 1e-8, 1 - 1e-8)
        u2c = np.clip(u2, 1e-8, 1 - 1e-8)
        x = (-np.log(u1c))**theta + (-np.log(u2c))**theta
        C = np.exp(-x**(1 / theta))
        log_c = (
            np.log(C)
            + (1 / theta - 2) * np.log(x)
            + (theta - 1) * (np.log(-np.log(u1c)) + np.log(-np.log(u2c)))
            - np.log(u1c) - np.log(u2c)
            + np.log(x**(1 / theta) + theta - 1)
        )
        ll = float(np.where(np.isfinite(log_c), log_c, -100).sum())
        tail_lower = 0.0
        tail_upper = 2 - 2 ** (1 / theta)

    elif family == "frank":
        u1c = np.clip(u1, 1e-8, 1 - 1e-8)
        u2c = np.clip(u2, 1e-8, 1 - 1e-8)
        e1 = np.exp(-theta * u1c)
        e2 = np.exp(-theta * u2c)
        e0 = np.exp(-theta) - 1
        denom = e0 + (e1 - 1) * (e2 - 1)
        log_c = (
            np.log(np.abs(theta))
            + np.log(np.abs(e0))
            + (-theta) * (u1c + u2c)
            - 2 * np.log(np.abs(np.where(np.abs(denom) > 1e-300, denom, 1e-300)))
        )
        ll = float(np.where(np.isfinite(log_c), log_c, -100).sum())
        tail_lower = 0.0
        tail_upper = 0.0
    else:
        raise ValueError(f"Unknown family: {family}")

    k = 1  # 1 tham số theta
    return CopulaResult(
        family=family,
        params={"theta": float(theta), "kendall_tau": float(tau)},
        log_lik=ll,
        aic=_aic(ll, k),
        bic=_bic(ll, k, n),
        tail_dep_lower=tail_lower,
        tail_dep_upper=tail_upper,
        n_obs=n,
    )


def sample_archimedean_copula(
    n: int,
    theta: float,
    family: str = "clayton",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sinh mẫu bivariate từ Archimedean copula dùng conditional sampling.

    Returns
    -------
    np.ndarray (n × 2)
    """
    rng = np.random.default_rng(seed)
    u1 = rng.uniform(0, 1, n)
    p = rng.uniform(0, 1, n)

    if family == "clayton":
        # Inverse conditional CDF: C_{2|1}(p|u1) = u1^{-θ-1}·(p^{-θ/(1+θ)} + u1^{-θ} - 1)^{-1-1/θ}
        u2 = (p**(-theta / (1 + theta)) + u1**(-theta) - 1) ** (-1 / theta)

    elif family == "gumbel":
        # Không có dạng giải tích đơn giản → dùng phép xấp xỉ
        # Sinh từ Fréchet marginals
        v = rng.uniform(0, 1, n)
        x1 = (-np.log(u1)) ** theta
        # Xấp xỉ: dùng log-stable biến
        alpha = 1 / theta
        w = -np.log(rng.uniform(0, 1, n))
        phi = np.pi * (rng.uniform(0, 1, n) - 0.5)
        stable = (
            np.sin(alpha * (phi + np.pi / 2)) ** (1 / alpha)
            / (np.cos(phi) ** (1 / alpha))
            * (np.cos(phi - alpha * (phi + np.pi / 2)) / w) ** ((1 - alpha) / alpha)
        )
        stable = np.abs(stable)
        x_sum = stable
        u1 = np.exp(-x1)
        u2 = np.exp(-(x_sum - x1))
        u2 = np.clip(u2, 1e-8, 1 - 1e-8)

    elif family == "frank":
        # Conditional: ln[1 + p(e^{-θ}-1) / (e^{-θu1}(p-1)-p)] / (-θ)
        e0 = np.exp(-theta) - 1
        num = p * e0
        den = np.exp(-theta * u1) * (p - 1) - p
        u2 = -np.log(1 + num / np.where(np.abs(den) > 1e-12, den, 1e-12)) / theta
        u2 = np.clip(u2, 1e-8, 1 - 1e-8)
    else:
        raise ValueError(f"Unknown family: {family}")

    u1 = np.clip(u1, 1e-8, 1 - 1e-8)
    u2 = np.clip(u2, 1e-8, 1 - 1e-8)
    return np.column_stack([u1, u2])


# ---------------------------------------------------------------------------
# Fit all copulas & select best
# ---------------------------------------------------------------------------

def fit_all_copulas(
    u: np.ndarray,
    rank_by: str = "aic",
) -> pd.DataFrame:
    """
    Fit nhiều copula và xếp hạng theo AIC/BIC.

    Parameters
    ----------
    u       : pseudo-observations (n × d)
    rank_by : 'aic' hoặc 'bic'

    Returns
    -------
    pd.DataFrame đã xếp hạng
    """
    results = []

    try:
        r = fit_gaussian_copula(u)
        results.append(r.summary())
    except Exception:
        pass

    try:
        r = fit_student_t_copula(u)
        results.append(r.summary())
    except Exception:
        pass

    if u.shape[1] == 2:
        for fam in ["clayton", "gumbel", "frank"]:
            try:
                r = fit_archimedean_copula(u, family=fam)
                results.append(r.summary())
            except Exception:
                pass

    if not results:
        raise RuntimeError("All copula fits failed.")

    df = pd.DataFrame(results).set_index("family")
    if rank_by in df.columns:
        df = df.sort_values(rank_by)
    return df


# ---------------------------------------------------------------------------
# Empirical tail dependence
# ---------------------------------------------------------------------------

def empirical_tail_dependence(
    u: np.ndarray,
    quantile: float = 0.05,
) -> Tuple[float, float]:
    """
    Ước lượng hệ số phụ thuộc đuôi thực nghiệm.

    λ_L = P(U₁ < q | U₂ < q) = P(U₁<q, U₂<q) / q
    λ_U = P(U₁ > 1-q | U₂ > 1-q) = P(U₁>1-q, U₂>1-q) / q

    Parameters
    ----------
    u        : pseudo-observations (n × 2)
    quantile : ngưỡng đuôi (default 0.05 = 5th percentile)

    Returns
    -------
    Tuple[float, float] – (lambda_lower, lambda_upper)
    """
    if u.shape[1] != 2:
        raise ValueError("Tail dependence requires bivariate (d=2) input.")

    u1, u2 = u[:, 0], u[:, 1]
    n = len(u1)

    # Lower tail
    lower_joint = np.mean((u1 <= quantile) & (u2 <= quantile))
    lambda_lower = float(lower_joint / quantile) if quantile > 0 else 0.0

    # Upper tail
    upper_joint = np.mean((u1 >= 1 - quantile) & (u2 >= 1 - quantile))
    lambda_upper = float(upper_joint / quantile) if quantile > 0 else 0.0

    return lambda_lower, lambda_upper


def tail_dependence_matrix(
    returns: np.ndarray,
    quantile: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tính ma trận hệ số phụ thuộc đuôi cho tất cả các cặp tài sản.

    Parameters
    ----------
    returns  : ma trận return (n_obs × d)
    quantile : ngưỡng đuôi

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame] – (lower_matrix, upper_matrix)
    """
    R = np.asarray(returns)
    n, d = R.shape
    u = pseudo_observations(R)

    lower = np.zeros((d, d))
    upper = np.zeros((d, d))
    np.fill_diagonal(lower, 1.0)
    np.fill_diagonal(upper, 1.0)

    for i in range(d):
        for j in range(i + 1, d):
            lam_l, lam_u = empirical_tail_dependence(
                u[:, [i, j]], quantile=quantile
            )
            lower[i, j] = lower[j, i] = lam_l
            upper[i, j] = upper[j, i] = lam_u

    return pd.DataFrame(lower), pd.DataFrame(upper)