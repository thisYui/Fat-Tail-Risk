"""
simulation/scenarios.py
-----------------------
Định nghĩa và sinh các kịch bản (scenarios) kinh tế vĩ mô / thị trường
dùng cho stress testing và phân tích tail risk:

  - ScenarioConfig  : dataclass chứa tham số của một kịch bản
  - Thư viện kịch bản lịch sử (Crisis library)
  - Scenario generator (parametric perturbation)
  - Factor shock scenarios
  - Bootstrap historical scenarios
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ScenarioConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """
    Chứa tham số đầy đủ của một kịch bản mô phỏng.

    Attributes
    ----------
    name        : tên kịch bản
    mu          : drift annualised
    sigma       : volatility annualised
    jump_intensity : cường độ jump (số jump/năm), 0 = không có jump
    jump_mu     : log-mean của jump size
    jump_sigma  : log-std của jump size
    dist        : phân phối innovation ('normal', 'student_t', 'skewed_t')
    dist_params : tham số thêm của phân phối (vd {'df': 3})
    description : mô tả ngắn về kịch bản
    """
    name: str
    mu: float = 0.0
    sigma: float = 0.20
    jump_intensity: float = 0.0
    jump_mu: float = -0.05
    jump_sigma: float = 0.08
    dist: str = "normal"
    dist_params: Dict = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# Crisis scenario library
# ---------------------------------------------------------------------------

CRISIS_SCENARIOS: Dict[str, ScenarioConfig] = {
    "base": ScenarioConfig(
        name="base",
        mu=0.07,
        sigma=0.16,
        description="Bình thường – thị trường bình ổn",
    ),
    "dot_com_2000": ScenarioConfig(
        name="dot_com_2000",
        mu=-0.30,
        sigma=0.35,
        jump_intensity=15,
        jump_mu=-0.04,
        jump_sigma=0.06,
        dist="student_t",
        dist_params={"df": 4},
        description="Dot-com bubble burst 2000-2002",
    ),
    "gfc_2008": ScenarioConfig(
        name="gfc_2008",
        mu=-0.45,
        sigma=0.60,
        jump_intensity=25,
        jump_mu=-0.06,
        jump_sigma=0.10,
        dist="student_t",
        dist_params={"df": 3},
        description="Global Financial Crisis 2008-2009",
    ),
    "covid_2020": ScenarioConfig(
        name="covid_2020",
        mu=-0.35,
        sigma=0.75,
        jump_intensity=30,
        jump_mu=-0.08,
        jump_sigma=0.12,
        dist="skewed_t",
        dist_params={"df": 4, "skew": -0.5},
        description="COVID-19 market crash Feb-Mar 2020",
    ),
    "flash_crash_2010": ScenarioConfig(
        name="flash_crash_2010",
        mu=0.05,
        sigma=0.25,
        jump_intensity=5,
        jump_mu=-0.10,
        jump_sigma=0.15,
        description="Flash Crash May 6, 2010",
    ),
    "taper_tantrum_2013": ScenarioConfig(
        name="taper_tantrum_2013",
        mu=-0.10,
        sigma=0.30,
        jump_intensity=8,
        jump_mu=-0.03,
        jump_sigma=0.05,
        description="Fed Taper Tantrum 2013",
    ),
    "ukraine_war_2022": ScenarioConfig(
        name="ukraine_war_2022",
        mu=-0.20,
        sigma=0.40,
        jump_intensity=12,
        jump_mu=-0.04,
        jump_sigma=0.07,
        dist="student_t",
        dist_params={"df": 5},
        description="Russia-Ukraine war market shock 2022",
    ),
    "stagflation": ScenarioConfig(
        name="stagflation",
        mu=-0.15,
        sigma=0.35,
        jump_intensity=6,
        jump_mu=-0.03,
        jump_sigma=0.06,
        description="Stagflation scenario (high inflation + recession)",
    ),
    "black_swan": ScenarioConfig(
        name="black_swan",
        mu=-0.60,
        sigma=0.90,
        jump_intensity=40,
        jump_mu=-0.10,
        jump_sigma=0.20,
        dist="student_t",
        dist_params={"df": 2},
        description="Extreme black swan event",
    ),
}


def get_scenario(name: str) -> ScenarioConfig:
    """
    Lấy kịch bản từ thư viện theo tên.

    Parameters
    ----------
    name : tên kịch bản (xem CRISIS_SCENARIOS.keys())

    Raises
    ------
    KeyError nếu tên không tồn tại
    """
    if name not in CRISIS_SCENARIOS:
        available = list(CRISIS_SCENARIOS.keys())
        raise KeyError(f"Scenario '{name}' not found. Available: {available}")
    return CRISIS_SCENARIOS[name]


def list_scenarios() -> pd.DataFrame:
    """
    Liệt kê tất cả các kịch bản có sẵn.

    Returns
    -------
    pd.DataFrame với cột: name, mu, sigma, jump_intensity, dist, description
    """
    rows = []
    for sc in CRISIS_SCENARIOS.values():
        rows.append(
            {
                "name": sc.name,
                "mu": sc.mu,
                "sigma": sc.sigma,
                "jump_intensity": sc.jump_intensity,
                "dist": sc.dist,
                "description": sc.description,
            }
        )
    return pd.DataFrame(rows).set_index("name")


# ---------------------------------------------------------------------------
# Parametric scenario perturbation
# ---------------------------------------------------------------------------

def perturb_scenario(
    base: ScenarioConfig,
    mu_shock: float = 0.0,
    sigma_multiplier: float = 1.0,
    jump_intensity_multiplier: float = 1.0,
    name_suffix: str = "_perturbed",
) -> ScenarioConfig:
    """
    Tạo kịch bản mới bằng cách biến đổi kịch bản gốc.

    Parameters
    ----------
    base                    : kịch bản gốc
    mu_shock                : cú sốc cộng vào drift (vd -0.10)
    sigma_multiplier        : nhân lên volatility (vd 2.0 = tăng gấp đôi vol)
    jump_intensity_multiplier : nhân lên tần suất jump

    Returns
    -------
    ScenarioConfig mới
    """
    from copy import deepcopy
    sc = deepcopy(base)
    sc.name = base.name + name_suffix
    sc.mu += mu_shock
    sc.sigma *= sigma_multiplier
    sc.jump_intensity *= jump_intensity_multiplier
    sc.description = f"Perturbed from '{base.name}'"
    return sc


def scenario_grid(
    base: ScenarioConfig,
    mu_shocks: Tuple[float, ...] = (-0.10, -0.20, -0.30),
    sigma_multipliers: Tuple[float, ...] = (1.5, 2.0, 3.0),
) -> List[ScenarioConfig]:
    """
    Tạo lưới kịch bản bằng cách kết hợp các mức shock.

    Parameters
    ----------
    mu_shocks        : các mức shock drift
    sigma_multipliers: các mức nhân volatility

    Returns
    -------
    List[ScenarioConfig] – (len(mu_shocks) × len(sigma_multipliers)) kịch bản
    """
    scenarios = []
    for dmu in mu_shocks:
        for smul in sigma_multipliers:
            suffix = f"_mu{dmu:+.2f}_sv{smul:.1f}"
            sc = perturb_scenario(
                base,
                mu_shock=dmu,
                sigma_multiplier=smul,
                name_suffix=suffix,
            )
            scenarios.append(sc)
    return scenarios


# ---------------------------------------------------------------------------
# Factor shock scenarios
# ---------------------------------------------------------------------------

@dataclass
class FactorShock:
    """
    Cú sốc trên các nhân tố rủi ro (risk factors).

    Attributes
    ----------
    name    : tên cú sốc
    shocks  : dict {factor_name: shock_value}
              vd {"equity": -0.30, "credit_spread": 0.05, "vix": 2.0}
    """
    name: str
    shocks: Dict[str, float]
    description: str = ""


FACTOR_SHOCKS: Dict[str, FactorShock] = {
    "equity_crash": FactorShock(
        name="equity_crash",
        shocks={"equity": -0.35, "vix": 3.0, "credit_spread": 0.04},
        description="Sập thị trường chứng khoán -35%",
    ),
    "rate_hike": FactorShock(
        name="rate_hike",
        shocks={"rates_10y": 0.02, "rates_2y": 0.03, "equity": -0.10},
        description="Fed tăng lãi suất mạnh +200bps",
    ),
    "credit_crisis": FactorShock(
        name="credit_crisis",
        shocks={"credit_spread": 0.08, "equity": -0.25, "liquidity": -0.5},
        description="Khủng hoảng tín dụng – spread tăng vọt",
    ),
    "fx_devaluation": FactorShock(
        name="fx_devaluation",
        shocks={"fx_usd": 0.20, "equity_em": -0.30, "commodity": -0.10},
        description="Phá giá đồng nội tệ 20%",
    ),
    "oil_spike": FactorShock(
        name="oil_spike",
        shocks={"oil": 0.50, "inflation": 0.03, "equity": -0.08},
        description="Giá dầu tăng 50% – cú sốc lạm phát",
    ),
}


def apply_factor_shock(
    portfolio_returns: pd.Series,
    factor_betas: Dict[str, float],
    shock: FactorShock,
) -> float:
    """
    Ước lượng tác động của một factor shock lên danh mục.

    P&L ≈ Σ (beta_i × shock_i)

    Parameters
    ----------
    portfolio_returns : chuỗi return lịch sử của danh mục
    factor_betas      : dict {factor_name: beta} của danh mục
    shock             : FactorShock cần áp dụng

    Returns
    -------
    float – ước tính P&L (có thể âm = lỗ)
    """
    pnl = 0.0
    for factor, shock_val in shock.shocks.items():
        beta = factor_betas.get(factor, 0.0)
        pnl += beta * shock_val
    return pnl


# ---------------------------------------------------------------------------
# Bootstrap historical scenarios
# ---------------------------------------------------------------------------

def bootstrap_scenarios(
    returns: pd.Series,
    n_scenarios: int = 1000,
    window: int = 20,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Tạo kịch bản bằng cách bootstrap từ chuỗi return lịch sử.

    Phương pháp: lấy mẫu ngẫu nhiên các cửa sổ `window` ngày từ lịch sử.

    Parameters
    ----------
    returns     : chuỗi return lịch sử
    n_scenarios : số kịch bản cần tạo
    window      : độ dài mỗi kịch bản (ngày)

    Returns
    -------
    pd.DataFrame shape (n_scenarios, window) – mỗi hàng là 1 kịch bản
    """
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    n = len(r)
    if n < window:
        raise ValueError(f"Series too short ({n}) for window={window}")

    max_start = n - window
    starts = rng.integers(0, max_start, size=n_scenarios)
    scenarios = np.array([r[s : s + window] for s in starts])

    cols = [f"day_{i+1}" for i in range(window)]
    return pd.DataFrame(scenarios, columns=cols)


def worst_scenarios(
    scenarios_df: pd.DataFrame,
    metric: str = "total_return",
    n: int = 10,
) -> pd.DataFrame:
    """
    Lọc ra n kịch bản tệ nhất theo một chỉ số.

    Parameters
    ----------
    scenarios_df : output của bootstrap_scenarios
    metric       : 'total_return' hoặc 'min_return'
    n            : số kịch bản cần lấy

    Returns
    -------
    pd.DataFrame – n hàng tệ nhất kèm giá trị metric
    """
    if metric == "total_return":
        scores = (1 + scenarios_df).prod(axis=1) - 1
    elif metric == "min_return":
        scores = scenarios_df.min(axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    df = scenarios_df.copy()
    df["_score"] = scores
    return df.nsmallest(n, "_score").drop(columns="_score")