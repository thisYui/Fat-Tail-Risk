"""
visualization/tail_plots.py
-----------------------------
Biểu đồ chuyên biệt cho phân tích đuôi phân phối (tail risk):

  - plot_var_cvar          : VaR/CVaR trên distribution
  - plot_evt_tail          : GPD fit lên đuôi thực nghiệm
  - plot_hill              : Hill estimator plot (tail index vs k)
  - plot_mean_excess       : Mean Excess Function
  - plot_threshold_stability: GPD tham số vs ngưỡng
  - plot_return_levels     : Return level plot (chu kỳ T năm)
  - plot_backtest_violations: VaR backtest với violations
  - plot_scenario_comparison: Fan chart nhiều kịch bản MC
  - plot_tail_dependence   : Scatter plot phụ thuộc đuôi
  - plot_copula_density    : Copula density contour
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from scipy import stats
from typing import Optional, Tuple, List, Dict, Union

from .plots import COLORS, _fig_ax, _save_or_show


# ---------------------------------------------------------------------------
# VaR / CVaR on distribution
# ---------------------------------------------------------------------------

def plot_var_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Histogram + VaR + CVaR với vùng tail tô màu.

    Vùng cam: giữa VaR và CVaR
    Vùng đỏ: phần loss vượt CVaR
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    alpha = 1 - confidence
    var = float(np.quantile(r, alpha))
    tail = r[r <= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var

    fig, ax = _fig_ax(figsize)

    # Histogram
    counts, bin_edges, patches = ax.hist(
        r, bins=80, density=True, color=COLORS["primary"], alpha=0.3, label="Return Distribution"
    )

    # Colour tail bins
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge < cvar:
            patch.set_facecolor(COLORS["secondary"])
            patch.set_alpha(0.8)
        elif left_edge < var:
            patch.set_facecolor(COLORS["warning"])
            patch.set_alpha(0.7)

    # KDE
    x = np.linspace(r.min() * 1.5, r.max() * 1.5, 400)
    kde = stats.gaussian_kde(r)
    ax.plot(x, kde(x), color=COLORS["primary"], linewidth=2, zorder=5)

    # Shade tail area under KDE
    x_tail = x[x <= var]
    ax.fill_between(x_tail, 0, kde(x_tail), alpha=0.35, color=COLORS["secondary"])

    # VaR / CVaR lines
    ax.axvline(var, color=COLORS["warning"], linewidth=2.5,
               label=f"VaR {int(confidence*100)}% = {var:.4f}", zorder=6)
    ax.axvline(cvar, color=COLORS["secondary"], linewidth=2.5, linestyle="--",
               label=f"CVaR {int(confidence*100)}% = {cvar:.4f}", zorder=6)

    ax.set_title(title or f"VaR & CVaR at {int(confidence*100)}% Confidence",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.9, fontsize=10)
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# GPD tail fit
# ---------------------------------------------------------------------------

def plot_evt_tail(
    returns: np.ndarray,
    threshold_quantile: float = 0.90,
    title: str = "EVT – GPD Tail Fit",
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Vẽ empirical tail vs GPD fitted tail (log-log scale).

    Parameters
    ----------
    threshold_quantile : ngưỡng POT
    """
    from ..models.evt import fit_gpd_pot

    r = np.asarray(returns)[~np.isnan(returns)]
    gpd = fit_gpd_pot(r, threshold_quantile=threshold_quantile)

    losses = -r[r < 0]
    u = gpd.threshold
    exceedances = losses[losses > u] - u
    n_total = len(losses)

    # Empirical survival function of exceedances
    sorted_exc = np.sort(exceedances)[::-1]
    empirical_sf = np.arange(1, len(sorted_exc) + 1) / n_total

    # GPD survival function
    x_grid = np.linspace(0, sorted_exc.max() * 1.1, 300)
    gpd_sf = (gpd.n_exceedances / n_total) * stats.genpareto.sf(
        x_grid, gpd.xi, loc=0, scale=gpd.sigma
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Log-log plot
    ax = axes[0]
    ax.scatter(np.log(sorted_exc), np.log(empirical_sf),
               s=12, color=COLORS["primary"], alpha=0.5, label="Empirical")
    ax.plot(np.log(x_grid[gpd_sf > 0] + u),
            np.log(gpd_sf[gpd_sf > 0]),
            color=COLORS["secondary"], linewidth=2,
            label=f"GPD(ξ={gpd.xi:.2f}, σ={gpd.sigma:.4f})")
    ax.set_xlabel("log(Loss)")
    ax.set_ylabel("log(Survival Probability)")
    ax.set_title("Log-Log Tail Plot", fontweight="bold")
    ax.legend(framealpha=0)

    # Right: Return level plot (log scale x)
    ax2 = axes[1]
    return_periods = np.logspace(0, 4, 200)
    trading_days = 252
    return_levels = []
    for T in return_periods:
        try:
            rl = gpd.return_level(T, trading_days=trading_days)
            return_levels.append(rl)
        except Exception:
            return_levels.append(np.nan)

    ax2.semilogx(return_periods, return_levels,
                 color=COLORS["secondary"], linewidth=2)
    ax2.set_xlabel("Return Period (years)")
    ax2.set_ylabel("Loss Level")
    ax2.set_title("Return Level Plot", fontweight="bold")
    ax2.axhline(gpd.threshold, color=COLORS["neutral"], linewidth=1,
                linestyle="--", label=f"Threshold u={gpd.threshold:.4f}")
    ax2.legend(framealpha=0)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Hill plot
# ---------------------------------------------------------------------------

def plot_hill(
    returns: np.ndarray,
    k_max_fraction: float = 0.3,
    title: str = "Hill Plot – Tail Index Estimator",
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Hill plot: tail index α vs số quan sát đuôi k.

    Vùng ổn định = ước lượng tốt của α.
    α nhỏ → đuôi dày hơn.
    """
    from ..features.tail import hill_plot as _hill_plot

    df = _hill_plot(returns, k_max_fraction=k_max_fraction)
    fig, ax = _fig_ax(figsize)

    ax.plot(df["k"], df["tail_index"], color=COLORS["primary"], linewidth=2)
    ax.fill_between(df["k"],
                    df["tail_index"] - 0.3,
                    df["tail_index"] + 0.3,
                    alpha=0.1, color=COLORS["primary"])
    ax.axhline(df["tail_index"].median(), color=COLORS["secondary"],
               linewidth=1.5, linestyle="--",
               label=f"Median α = {df['tail_index'].median():.2f}")

    # Shade normal zone
    ax.axhline(2, color=COLORS["warning"], linewidth=1, linestyle=":",
               label="α=2 (Normal threshold)")

    ax.set_xlabel("k (Number of tail observations)")
    ax.set_ylabel("Tail Index α")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(framealpha=0)
    ax.set_ylim(0, min(df["tail_index"].max() * 1.5, 15))
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Mean Excess Function
# ---------------------------------------------------------------------------

def plot_mean_excess(
    returns: np.ndarray,
    title: str = "Mean Excess Function",
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Vẽ Mean Excess Function (MEF) để chọn ngưỡng POT.

    Đường MEF tuyến tính và tăng dần tại u → GPD phù hợp.
    """
    from ..models.evt import mean_excess_plot_data

    df = mean_excess_plot_data(returns)
    fig, ax = _fig_ax(figsize)

    ax.plot(df["threshold"], df["mean_excess"],
            color=COLORS["primary"], linewidth=2, label="MEF")
    ax.fill_between(df["threshold"],
                    df["mean_excess"] - df["mean_excess"].std() * 0.3,
                    df["mean_excess"] + df["mean_excess"].std() * 0.3,
                    alpha=0.1, color=COLORS["primary"])

    # Annotate n_exceedances on secondary axis
    ax2 = ax.twinx()
    ax2.plot(df["threshold"], df["n_exceedances"],
             color=COLORS["neutral"], linewidth=1, linestyle="--", alpha=0.6)
    ax2.set_ylabel("N Exceedances", color=COLORS["neutral"], fontsize=9)

    ax.set_xlabel("Threshold u")
    ax.set_ylabel("Mean Excess e(u)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(framealpha=0)
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Threshold stability
# ---------------------------------------------------------------------------

def plot_threshold_stability(
    returns: np.ndarray,
    title: str = "GPD Parameter Stability",
    figsize: Tuple[float, float] = (14, 5),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Vẽ ξ và σ* theo ngưỡng u (threshold stability plot).

    Chọn ngưỡng tối thiểu nơi cả ξ và σ* ổn định.
    """
    from ..models.evt import threshold_stability_plot_data

    df = threshold_stability_plot_data(returns)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, col, label in [
        (axes[0], "xi", "Shape (ξ)"),
        (axes[1], "sigma_star", "Modified Scale (σ*)"),
    ]:
        ax.plot(df["threshold"], df[col], color=COLORS["primary"], linewidth=2)
        ax.axhline(df[col].median(), color=COLORS["secondary"],
                   linewidth=1.5, linestyle="--",
                   label=f"Median = {df[col].median():.3f}")
        ax.set_xlabel("Threshold u")
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")
        ax.legend(framealpha=0)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# VaR backtest violations
# ---------------------------------------------------------------------------

def plot_backtest_violations(
    returns: pd.Series,
    var_forecasts: pd.Series,
    model_name: str = "VaR Model",
    confidence: float = 0.95,
    figsize: Tuple[float, float] = (14, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Vẽ chuỗi return cùng VaR forecast và đánh dấu violations.

    Parameters
    ----------
    returns       : chuỗi return thực tế
    var_forecasts : chuỗi VaR forecast (âm)
    """
    r = returns.dropna()
    var = var_forecasts.reindex(r.index)
    violations = r < var

    fig, ax = _fig_ax(figsize)
    idx = r.index

    # Return bars
    pos = r.clip(lower=0)
    neg = r.clip(upper=0)
    ax.bar(idx, pos.values, color=COLORS["accent"], alpha=0.5, width=1)
    ax.bar(idx, neg.values, color=COLORS["neutral"], alpha=0.5, width=1)

    # VaR line
    ax.plot(idx, var.values, color=COLORS["secondary"],
            linewidth=1.5, label=f"VaR {int(confidence*100)}%", zorder=4)

    # Violations
    viol_idx = r.index[violations]
    ax.scatter(viol_idx, r[violations].values,
               color=COLORS["secondary"], zorder=5, s=40,
               label=f"Violations (n={violations.sum()}, rate={violations.mean():.2%})")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"VaR Backtest – {model_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Return")
    ax.legend(framealpha=0.9)
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Monte Carlo scenario fan chart
# ---------------------------------------------------------------------------

def plot_scenario_fan(
    paths: np.ndarray,
    S0: float = 100.0,
    percentiles: Tuple[float, ...] = (5, 25, 50, 75, 95),
    n_sample_paths: int = 50,
    title: str = "Monte Carlo Scenario Fan Chart",
    figsize: Tuple[float, float] = (14, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Fan chart cho Monte Carlo paths.

    Parameters
    ----------
    paths           : (n_steps+1, n_paths) price paths
    percentiles     : các phân vị cần vẽ
    n_sample_paths  : số đường ngẫu nhiên vẽ nền
    """
    n_steps, n_paths = paths.shape[0] - 1, paths.shape[1]
    t = np.arange(n_steps + 1)

    fig, ax = _fig_ax(figsize)

    # Sample paths background
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_paths, size=min(n_sample_paths, n_paths), replace=False)
    for idx in sample_idx:
        ax.plot(t, paths[:, idx], color=COLORS["primary"], alpha=0.05, linewidth=0.8)

    # Percentile bands
    pct_vals = np.percentile(paths, percentiles, axis=1)
    band_colors = ["#FED7D7", "#FEB2B2", "#FC8181", "#FEB2B2", "#FED7D7"]
    pct_arr = list(percentiles)

    for i in range(len(pct_arr) // 2):
        lo_pct = pct_vals[i]
        hi_pct = pct_vals[-(i + 1)]
        ax.fill_between(t, lo_pct, hi_pct,
                        alpha=0.25, color=COLORS["secondary"],
                        label=f"P{pct_arr[i]}–P{pct_arr[-(i+1)]}")

    # Median
    median = np.percentile(paths, 50, axis=1)
    ax.plot(t, median, color=COLORS["secondary"], linewidth=2.5, label="Median", zorder=5)

    # Starting value
    ax.axhline(S0, color="black", linewidth=0.8, linestyle="--")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price / Value")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), framealpha=0, fontsize=9)
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Tail dependence scatter
# ---------------------------------------------------------------------------

def plot_tail_dependence(
    returns_df: pd.DataFrame,
    asset1: str,
    asset2: str,
    quantile: float = 0.05,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Scatter plot thể hiện phụ thuộc đuôi giữa hai tài sản.

    Tô màu đỏ: cùng nằm trong đuôi trái (joint lower tail events).

    Parameters
    ----------
    quantile : ngưỡng đuôi (default 5th percentile)
    """
    r1 = returns_df[asset1].dropna()
    r2 = returns_df[asset2].dropna()
    common = r1.index.intersection(r2.index)
    r1, r2 = r1.loc[common], r2.loc[common]

    q1 = float(np.quantile(r1, quantile))
    q2 = float(np.quantile(r2, quantile))

    in_lower_tail = (r1 <= q1) & (r2 <= q2)
    in_upper_tail = (r1 >= np.quantile(r1, 1 - quantile)) & \
                    (r2 >= np.quantile(r2, 1 - quantile))

    fig, ax = _fig_ax(figsize)

    # Normal points
    mask_normal = ~in_lower_tail & ~in_upper_tail
    ax.scatter(r1[mask_normal], r2[mask_normal],
               color=COLORS["neutral"], alpha=0.3, s=8, label="Normal")

    # Lower tail joint events
    ax.scatter(r1[in_lower_tail], r2[in_lower_tail],
               color=COLORS["secondary"], alpha=0.8, s=25,
               label=f"Lower Tail (n={in_lower_tail.sum()})")

    # Upper tail joint events
    ax.scatter(r1[in_upper_tail], r2[in_upper_tail],
               color=COLORS["accent"], alpha=0.8, s=25,
               label=f"Upper Tail (n={in_upper_tail.sum()})")

    # Threshold lines
    ax.axvline(q1, color=COLORS["secondary"], linewidth=1.2, linestyle="--", alpha=0.6)
    ax.axhline(q2, color=COLORS["secondary"], linewidth=1.2, linestyle="--", alpha=0.6)

    # Empirical tail dependence
    lambda_lower = float(np.mean(in_lower_tail)) / quantile if quantile > 0 else 0
    lambda_upper = float(np.mean(in_upper_tail)) / quantile if quantile > 0 else 0

    ax.set_xlabel(asset1)
    ax.set_ylabel(asset2)
    ax.set_title(
        title or f"Tail Dependence: {asset1} vs {asset2}\n"
                 f"λ_lower={lambda_lower:.3f}  λ_upper={lambda_upper:.3f}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(framealpha=0.9, fontsize=9)
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Scenario comparison bar chart
# ---------------------------------------------------------------------------

def plot_scenario_comparison(
    scenario_summary: pd.DataFrame,
    metric: str = "cvar_95",
    title: str = "Scenario Comparison",
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Vẽ bar chart so sánh VaR/CVaR các kịch bản stress.

    Parameters
    ----------
    scenario_summary : output của compare_scenarios_summary()
    metric           : cột cần vẽ
    """
    df = scenario_summary[[metric]].dropna().sort_values(metric)
    fig, ax = _fig_ax(figsize)

    colors = [
        COLORS["secondary"] if v < -0.15 else
        COLORS["warning"] if v < -0.05 else
        COLORS["accent"]
        for v in df[metric]
    ]
    bars = ax.barh(df.index, df[metric], color=colors, alpha=0.8)

    for bar, val in zip(bars, df[metric]):
        ax.text(val, bar.get_y() + bar.get_height() / 2,
                f" {val:.3f}", va="center", fontsize=9)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(metric.upper().replace("_", " "))
    ax.set_title(f"{title} – {metric.upper()}", fontsize=14, fontweight="bold")
    return _save_or_show(fig, save_path)