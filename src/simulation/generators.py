"""
simulation/generators.py
------------------------
Các bộ sinh số ngẫu nhiên cơ bản (primitive generators) cho các
phân phối fat-tail thường dùng trong tài chính:

  - Normal / Multivariate Normal
  - Student-t (fat tail)
  - Skewed Student-t (Hansen 1994)
  - Stable distribution (α-stable)
  - Pareto / Generalized Pareto (GPD)
  - Laplace
  - Mixture of Normals (MoN)

Tất cả hàm đều nhận `seed` tùy chọn để tái lập kết quả.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------

def normal(
    n: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sinh n mẫu từ phân phối Normal N(mu, sigma²)."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mu, scale=sigma, size=n)


def multivariate_normal(
    n: int,
    mu: np.ndarray,
    cov: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sinh n mẫu từ phân phối Normal đa chiều.

    Parameters
    ----------
    mu  : vector kỳ vọng (d,)
    cov : ma trận hiệp phương sai (d, d)

    Returns
    -------
    np.ndarray shape (n, d)
    """
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=mu, cov=cov, size=n)


# ---------------------------------------------------------------------------
# Student-t
# ---------------------------------------------------------------------------

def student_t(
    n: int,
    df: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sinh n mẫu từ Student-t với bậc tự do `df`.

    df nhỏ → đuôi dày hơn (df=3 rất fat-tail, df→∞ → Normal).

    Parameters
    ----------
    df    : degrees of freedom (> 0). Khuyến nghị df > 2 để variance tồn tại.
    mu    : location
    sigma : scale (> 0)

    Raises
    ------
    ValueError
        Nếu df ≤ 0 hoặc sigma ≤ 0.
    """
    if df <= 0:
        raise ValueError(f"df phải > 0, nhận được df={df}.")
    if sigma <= 0:
        raise ValueError(f"sigma phải > 0, nhận được sigma={sigma}.")
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df=df, size=n)
    return mu + sigma * z


def multivariate_student_t(
    n: int,
    df: float,
    mu: np.ndarray,
    cov: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sinh n mẫu từ Student-t đa chiều.

    Parameters
    ----------
    df  : degrees of freedom
    mu  : vector location (d,)
    cov : ma trận scale (d, d)

    Returns
    -------
    np.ndarray shape (n, d)
    """
    rng = np.random.default_rng(seed)
    d = len(mu)
    # X = Z / sqrt(V/df), Z ~ MVN(0, cov), V ~ Chi2(df)
    z = rng.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)
    v = rng.chisquare(df=df, size=(n, 1))
    return mu + z / np.sqrt(v / df)


# ---------------------------------------------------------------------------
# Skewed Student-t (Hansen 1994)
# ---------------------------------------------------------------------------

def skewed_student_t(
    n: int,
    df: float,
    skew: float = 0.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sinh n mẫu từ Skewed Student-t theo phương pháp Fernandez & Steel (1998).

    Kết hợp fat-tail (df) và bất đối xứng (skew) bằng cách scale riêng
    hai nhánh của phân phối t-Student chuẩn.

    Parameters
    ----------
    df   : degrees of freedom (> 2, bắt buộc để moment bậc 2 tồn tại)
    skew : log-scale asymmetry parameter (thực số bất kỳ; default 0 = symmetric).
           skew < 0 → lệch trái (downside tail nặng hơn — phổ biến với tài chính)
           skew > 0 → lệch phải
           |skew| nhỏ (< 1) cho phân phối có ý nghĩa thực tế nhất.
    mu    : location
    sigma : scale (> 0)

    Notes
    -----
    **Không phải** Hansen (1994) parameterisation (lambda ∈ (−1, 1)).
    Phương pháp này dùng gamma = exp(skew) để encode asymmetry:
      - gamma > 1 (skew > 0) → nhánh phải scale lớn hơn → lệch phải
      - gamma < 1 (skew < 0) → nhánh trái scale nhỏ hơn → lệch trái
      - gamma = 1 (skew = 0) → phân phối t-Student đối xứng

    References
    ----------
    Fernandez, C. & Steel, M. F. J. (1998). On Bayesian modeling of fat tails
    and skewness. JASA, 93(441), 359–371.
    """
    if df <= 2:
        raise ValueError(f"df phải > 2 để variance tồn tại, nhận được df={df}.")
    if sigma <= 0:
        raise ValueError(f"sigma phải > 0, nhận được sigma={sigma}.")

    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, size=n)

    # gamma = exp(skew): hệ số scale bất đối xứng
    # gamma > 1 → nhánh phải rộng hơn; gamma < 1 → nhánh trái rộng hơn
    gamma = np.exp(skew)
    # cut = điểm phân chia xác suất: P(nhánh trái) = cut
    cut = gamma / (gamma + 1.0 / gamma)

    left_mask = u < cut
    samples = np.empty(n)

    # Nhánh trái (u < cut): scale = 1/gamma → đuôi trái hẹp hơn khi gamma > 1
    p_left = u[left_mask] / cut
    samples[left_mask] = stats.t.ppf(p_left / 2, df=df) / gamma

    # Nhánh phải (u >= cut): scale = gamma → đuôi phải rộng hơn khi gamma > 1
    p_right = (u[~left_mask] - cut) / (1.0 - cut)
    samples[~left_mask] = stats.t.ppf(0.5 + p_right / 2, df=df) * gamma

    return mu + sigma * samples


# ---------------------------------------------------------------------------
# Alpha-Stable
# ---------------------------------------------------------------------------

def alpha_stable(
    n: int,
    alpha: float = 1.7,
    beta: float = 0.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sinh n mẫu từ phân phối α-Stable (Lévy stable).

    Phân phối này có đuôi power-law khi α < 2.

    Parameters
    ----------
    alpha : stability index ∈ (0, 2].
            alpha=2 → Normal; alpha=1 → Cauchy; alpha<2 → fat-tail
    beta  : skewness parameter ∈ [-1, 1]. 0 = symmetric
    mu    : location (shift)
    sigma : scale

    Notes
    -----
    Dùng phương pháp Chambers-Mallows-Stuck (CMS 1976).
    """
    rng = np.random.default_rng(seed)
    # CMS method
    U = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    W = rng.exponential(scale=1.0, size=n)

    if np.isclose(alpha, 1.0):
        # Cauchy case
        X = (
            (2 / np.pi)
            * ((np.pi / 2 + beta * U) * np.tan(U)
               - beta * np.log(np.pi / 2 * W * np.cos(U) / (np.pi / 2 + beta * U)))
        )
    else:
        zeta = -beta * np.tan(np.pi * alpha / 2)
        xi = (1 / alpha) * np.arctan(-zeta)
        X = (
            (1 + zeta**2) ** (1 / (2 * alpha))
            * np.sin(alpha * (U + xi))
            / (np.cos(U) ** (1 / alpha))
            * (np.cos(U - alpha * (U + xi)) / W) ** ((1 - alpha) / alpha)
        )

    return mu + sigma * X


# ---------------------------------------------------------------------------
# Generalized Pareto Distribution (GPD)
# ---------------------------------------------------------------------------

def generalized_pareto(
    n: int,
    xi: float = 0.3,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sinh n mẫu từ Generalized Pareto Distribution (GPD) bằng inverse-CDF.

    Dùng trong mô hình Peaks-Over-Threshold (POT/EVT).

    Parameters
    ----------
    xi    : shape parameter (tail index).
            xi > 0 → Pareto (fat-tail, heavy).
            xi = 0 → Exponential (handled via limit).
            xi < 0 → Bounded distribution (light tail).
    mu    : location — thường là threshold u trong mô hình POT.
    sigma : scale (> 0).

    Raises
    ------
    ValueError
        Nếu sigma ≤ 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma phải > 0, nhận được sigma={sigma}.")
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, size=n)
    if np.isclose(xi, 0.0):
        # Giới hạn xi → 0: GPD → Exponential(scale=sigma)
        return mu - sigma * np.log(u)
    return mu + sigma * (u ** (-xi) - 1) / xi


# ---------------------------------------------------------------------------
# Laplace
# ---------------------------------------------------------------------------

def laplace(
    n: int,
    mu: float = 0.0,
    b: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sinh n mẫu từ phân phối Laplace (Double Exponential).

    Đuôi dày hơn Normal nhưng nhẹ hơn Student-t.

    Parameters
    ----------
    mu : location (mean)
    b  : scale (b = std / sqrt(2))
    """
    rng = np.random.default_rng(seed)
    return rng.laplace(loc=mu, scale=b, size=n)


# ---------------------------------------------------------------------------
# Mixture of Normals (MoN)
# ---------------------------------------------------------------------------

def mixture_of_normals(
    n: int,
    weights: Tuple[float, ...] = (0.9, 0.1),
    mus: Tuple[float, ...] = (0.0, 0.0),
    sigmas: Tuple[float, ...] = (1.0, 5.0),
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sinh n mẫu từ Mixture of Normals.

    Rất phổ biến để mô phỏng return với occasional large jumps.

    Parameters
    ----------
    weights : trọng số của từng component (tổng = 1)
    mus     : vector kỳ vọng của từng component
    sigmas  : vector độ lệch chuẩn của từng component

    Example
    -------
    weights=(0.9, 0.1), mus=(0, 0), sigmas=(1, 5):
    → 90% Normal thường + 10% Normal biến động lớn
    """
    rng = np.random.default_rng(seed)
    weights = np.array(weights)
    weights = weights / weights.sum()
    components = rng.choice(len(weights), size=n, p=weights)
    samples = np.empty(n)
    for i, (m, s) in enumerate(zip(mus, sigmas)):
        mask = components == i
        samples[mask] = rng.normal(loc=m, scale=s, size=mask.sum())
    return samples


# ---------------------------------------------------------------------------
# Convenience: sample from named distribution
# ---------------------------------------------------------------------------

DISTRIBUTION_MAP = {
    "normal": normal,
    "student_t": student_t,
    "skewed_t": skewed_student_t,
    "alpha_stable": alpha_stable,
    "gpd": generalized_pareto,
    "laplace": laplace,
    "mixture_normal": mixture_of_normals,
}


def sample(
    dist: str,
    n: int,
    seed: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """
    Giao diện thống nhất để sinh mẫu từ bất kỳ phân phối nào.

    Parameters
    ----------
    dist : tên phân phối, một trong:
           'normal', 'student_t', 'skewed_t', 'alpha_stable',
           'gpd', 'laplace', 'mixture_normal'
    n    : số mẫu
    seed : random seed
    **kwargs : tham số riêng của từng phân phối

    Example
    -------
    >>> x = sample('student_t', n=1000, df=3, seed=42)
    """
    if dist not in DISTRIBUTION_MAP:
        raise ValueError(
            f"Unknown distribution '{dist}'. "
            f"Available: {list(DISTRIBUTION_MAP.keys())}"
        )
    fn = DISTRIBUTION_MAP[dist]
    return fn(n=n, seed=seed, **kwargs)