"""
visualization/plots.py
-----------------------
Biểu đồ tổng quát cho phân tích return và rủi ro:

  - plot_returns          : chuỗi return theo thời gian
  - plot_cumulative       : cumulative return (growth of $1)
  - plot_drawdown         : drawdown waterfall
  - plot_rolling_vol      : rolling volatility
  - plot_return_dist      : histogram + KDE của return
  - plot_qq               : Q-Q plot so với Normal / Student-t
  - plot_acf_pacf         : autocorrelation (return & return²)
  - plot_risk_comparison  : bar chart so sánh chỉ số rủi ro
  - plot_correlation      : heatmap correlation matrix
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy import stats
from typing import Optional, Union, Tuple, List, Dict

# Matplotlib style mặc định cho toàn package
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

COLORS = {
    "primary": "#2B6CB0",
    "secondary": "#E53E3E",
    "accent": "#38A169",
    "neutral": "#718096",
    "warning": "#D69E2E",
    "light": "#EBF4FF",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_ax(
    figsize: Tuple[float, float] = (12, 5),
    nrows: int = 1,
    ncols: int = 1,
    **kwargs,
) -> Tuple[Figure, Union[Axes, np.ndarray]]:
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, ax


def _save_or_show(fig: Figure, save_path: Optional[str] = None) -> Figure:
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Return time series
# ---------------------------------------------------------------------------

def plot_returns(
    returns: Union[pd.Series, pd.DataFrame],
    title: str = "Daily Returns",
    figsize: Tuple[float, float] = (14, 4),
    color: str = COLORS["primary"],
    save_path: Optional[str] = None,
) -> Figure:
    """
    Vẽ chuỗi return theo thời gian (bar chart).

    Parameters
    ----------
    returns  : pd.Series hoặc DataFrame nhiều tài sản
    """
    fig, ax = _fig_ax(figsize)

    if isinstance(returns, pd.Series):
        pos = returns.clip(lower=0)
        neg = returns.clip(upper=0)
        idx = range(len(returns)) if not hasattr(returns.index, "freq") else returns.index
        ax.bar(idx, pos, color=COLORS["accent"], alpha=0.7, label="Positive")
        ax.bar(idx, neg, color=COLORS["secondary"], alpha=0.7, label="Negative")
        ax.axhline(0, color="black", linewidth=0.8)
    else:
        for col in returns.columns:
            ax.plot(returns.index, returns[col], linewidth=1.2, label=col)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Return")
    ax.legend(framealpha=0)
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Cumulative return
# ---------------------------------------------------------------------------

def plot_cumulative(
    returns: Union[pd.Series, pd.DataFrame],
    title: str = "Cumulative Return (Growth of $1)",
    figsize: Tuple[float, float] = (14, 5),
    log_scale: bool = False,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Vẽ cumulative return.

    Parameters
    ----------
    log_scale : True → trục y log scale
    """
    fig, ax = _fig_ax(figsize)

    if isinstance(returns, pd.Series):
        cum = (1 + returns.fillna(0)).cumprod()
        ax.plot(cum.index if hasattr(cum.index, "freq") else range(len(cum)),
                cum.values, color=COLORS["primary"], linewidth=2)
        ax.fill_between(
            cum.index if hasattr(cum.index, "freq") else range(len(cum)),
            1, cum.values,
            where=cum.values >= 1, alpha=0.15, color=COLORS["accent"],
        )
        ax.fill_between(
            cum.index if hasattr(cum.index, "freq") else range(len(cum)),
            1, cum.values,
            where=cum.values < 1, alpha=0.15, color=COLORS["secondary"],
        )
    else:
        cum = (1 + returns.fillna(0)).cumprod()
        for col in cum.columns:
            ax.plot(cum.index, cum[col], linewidth=2, label=col)
        ax.legend(framealpha=0)

    ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value")
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def plot_drawdown(
    returns: pd.Series,
    title: str = "Drawdown",
    figsize: Tuple[float, float] = (14, 4),
    save_path: Optional[str] = None,
) -> Figure:
    """Vẽ drawdown waterfall chart."""
    fig, ax = _fig_ax(figsize)

    cum = (1 + returns.fillna(0)).cumprod()
    hwm = cum.cummax()
    dd = (cum - hwm) / hwm

    idx = dd.index if hasattr(dd.index, "freq") else range(len(dd))
    ax.fill_between(idx, 0, dd.values, color=COLORS["secondary"], alpha=0.6)
    ax.plot(idx, dd.values, color=COLORS["secondary"], linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)

    mdd = float(dd.min())
    mdd_idx = dd.idxmin()
    ax.annotate(
        f"Max DD: {mdd:.1%}",
        xy=(mdd_idx, mdd),
        xytext=(mdd_idx, mdd - 0.02),
        fontsize=9, color=COLORS["secondary"],
        arrowprops=dict(arrowstyle="->", color=COLORS["secondary"]),
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Rolling volatility
# ---------------------------------------------------------------------------

def plot_rolling_vol(
    returns: pd.Series,
    windows: List[int] = [21, 63, 252],
    freq: int = 252,
    title: str = "Rolling Annualised Volatility",
    figsize: Tuple[float, float] = (14, 4),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Vẽ rolling annualised volatility với nhiều cửa sổ.

    Parameters
    ----------
    windows : danh sách cửa sổ (ngày) cần vẽ
    freq    : số ngày / năm để annualise
    """
    fig, ax = _fig_ax(figsize)
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["warning"]]
    labels = {21: "1-Month", 63: "1-Quarter", 126: "6-Month", 252: "1-Year"}

    for i, w in enumerate(windows):
        rv = returns.rolling(w).std() * np.sqrt(freq)
        label = labels.get(w, f"{w}d")
        idx = rv.index if hasattr(rv.index, "freq") else range(len(rv))
        ax.plot(idx, rv.values, linewidth=1.5, label=label, color=colors[i % len(colors)])

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylabel("Annualised Volatility")
    ax.legend(framealpha=0)
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Return distribution
# ---------------------------------------------------------------------------

def plot_return_dist(
    returns: np.ndarray,
    title: str = "Return Distribution",
    bins: int = 80,
    fit_normal: bool = True,
    fit_t: bool = True,
    var_levels: Tuple[float, ...] = (0.95, 0.99),
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Histogram return + KDE + fitted distributions + VaR lines.

    Parameters
    ----------
    fit_normal : overlay Normal fit
    fit_t      : overlay Student-t fit
    var_levels : vẽ đường VaR dọc
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    fig, ax = _fig_ax(figsize)

    # Histogram
    ax.hist(r, bins=bins, density=True, color=COLORS["primary"],
            alpha=0.4, label="Empirical", zorder=2)

    x = np.linspace(r.min() * 1.5, r.max() * 1.5, 500)

    # KDE
    kde = stats.gaussian_kde(r)
    ax.plot(x, kde(x), color=COLORS["primary"], linewidth=2, label="KDE", zorder=3)

    # Normal fit
    if fit_normal:
        mu, sigma = r.mean(), r.std()
        ax.plot(x, stats.norm.pdf(x, mu, sigma),
                color=COLORS["accent"], linewidth=1.8, linestyle="--",
                label=f"Normal(μ={mu:.4f}, σ={sigma:.4f})", zorder=4)

    # Student-t fit
    if fit_t:
        df_t, loc_t, scale_t = stats.t.fit(r)
        ax.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t),
                color=COLORS["warning"], linewidth=1.8, linestyle="-.",
                label=f"Student-t(df={df_t:.1f})", zorder=4)

    # VaR lines
    line_colors = [COLORS["secondary"], "#7B2D8B"]
    for i, cl in enumerate(var_levels):
        var_val = float(np.quantile(r, 1 - cl))
        ax.axvline(var_val, color=line_colors[i % 2], linewidth=1.5,
                   linestyle=":", label=f"VaR {int(cl*100)}%={var_val:.4f}")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0, fontsize=9)
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Q-Q plot
# ---------------------------------------------------------------------------

def plot_qq(
    returns: np.ndarray,
    dist: str = "norm",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Quantile-Quantile plot so với phân phối lý thuyết.

    Parameters
    ----------
    dist : 'norm' (Normal) hoặc 't' (Student-t)
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    fig, ax = _fig_ax(figsize)

    if dist == "norm":
        (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm", fit=True)
        dist_label = "Normal"
    else:
        df_t, loc_t, scale_t = stats.t.fit(r)
        (osm, osr), (slope, intercept, _) = stats.probplot(
            r, dist=stats.t, sparams=(df_t, loc_t, scale_t), fit=True
        )
        dist_label = f"Student-t(df={df_t:.1f})"

    ax.scatter(osm, osr, color=COLORS["primary"], s=10, alpha=0.6, label="Sample quantiles")
    line_x = np.array([osm.min(), osm.max()])
    ax.plot(line_x, slope * line_x + intercept,
            color=COLORS["secondary"], linewidth=2, label="Reference line")

    ax.set_title(title or f"Q-Q Plot vs {dist_label}", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Theoretical Quantiles ({dist_label})")
    ax.set_ylabel("Sample Quantiles")
    ax.legend(framealpha=0)
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# ACF / PACF
# ---------------------------------------------------------------------------

def plot_acf_pacf(
    returns: np.ndarray,
    lags: int = 40,
    title_prefix: str = "",
    figsize: Tuple[float, float] = (14, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """
    ACF và PACF của returns và returns² (ARCH effects).

    Returns
    -------
    Figure với 4 subplots: ACF(r), PACF(r), ACF(r²), PACF(r²)
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    r = pd.Series(returns).dropna().values
    r2 = r ** 2

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    conf_int = 1.96 / np.sqrt(len(r))

    for ax, data, sq, fn_name in [
        (axes[0, 0], r, False, "ACF"),
        (axes[0, 1], r, False, "PACF"),
        (axes[1, 0], r2, True, "ACF"),
        (axes[1, 1], r2, True, "PACF"),
    ]:
        label = f"{'Squared ' if sq else ''}Returns – {fn_name}"
        if fn_name == "ACF":
            plot_acf(data, lags=lags, ax=ax, title=label, alpha=0.05)
        else:
            plot_pacf(data, lags=lags, ax=ax, title=label, alpha=0.05, method="ywm")
        ax.axhline(conf_int, color=COLORS["warning"], linewidth=1, linestyle="--")
        ax.axhline(-conf_int, color=COLORS["warning"], linewidth=1, linestyle="--")

    if title_prefix:
        fig.suptitle(f"{title_prefix} – ACF/PACF Analysis", fontsize=14, fontweight="bold")

    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Risk metric comparison bar chart
# ---------------------------------------------------------------------------

def plot_risk_comparison(
    risk_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    title: str = "Risk Metrics Comparison",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Bar chart so sánh chỉ số rủi ro giữa nhiều tài sản / chiến lược.

    Parameters
    ----------
    risk_df  : DataFrame (columns = assets, rows = metrics)
    metrics  : danh sách metrics cần vẽ (None = tất cả)
    """
    if metrics is None:
        metrics = list(risk_df.index)

    n_metrics = len(metrics)
    n_assets = len(risk_df.columns)
    if figsize is None:
        figsize = (max(12, n_metrics * 2), max(5, n_assets * 1.5))

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
              COLORS["warning"], COLORS["neutral"]]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        if metric not in risk_df.index:
            ax.set_visible(False)
            continue
        vals = risk_df.loc[metric]
        bars = ax.barh(vals.index, vals.values,
                       color=[colors[j % len(colors)] for j in range(len(vals))],
                       alpha=0.8)
        ax.set_title(metric.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.6)

        # Value labels
        for bar, val in zip(bars, vals.values):
            if not np.isnan(val):
                ax.text(val, bar.get_y() + bar.get_height() / 2,
                        f" {val:.3f}", va="center", fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation(
    returns_df: pd.DataFrame,
    method: str = "pearson",
    title: str = "Correlation Matrix",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Heatmap ma trận tương quan.

    Parameters
    ----------
    method : 'pearson', 'spearman', hoặc 'kendall'
    """
    corr = returns_df.corr(method=method)
    n = len(corr)
    if figsize is None:
        figsize = (max(8, n), max(7, n - 1))

    fig, ax = _fig_ax(figsize)
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    ax.set_title(f"{title} ({method.capitalize()})", fontsize=14, fontweight="bold")
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Performance dashboard (multi-panel)
# ---------------------------------------------------------------------------

def plot_performance_dashboard(
    returns: pd.Series,
    title: str = "Performance Dashboard",
    risk_free: float = 0.0,
    freq: int = 252,
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Dashboard tổng hợp: cumulative return, drawdown, rolling vol, distribution.

    Parameters
    ----------
    returns   : chuỗi return có DatetimeIndex
    risk_free : annualised risk-free rate
    """
    r = returns.dropna()
    cum = (1 + r).cumprod()
    hwm = cum.cummax()
    dd = (cum - hwm) / hwm
    rv_21 = r.rolling(21).std() * np.sqrt(freq)
    rv_63 = r.rolling(63).std() * np.sqrt(freq)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    idx = r.index if hasattr(r.index, "freq") else range(len(r))

    # 1. Cumulative return
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(idx, cum.values, color=COLORS["primary"], linewidth=2)
    ax1.fill_between(idx, 1, cum.values,
                     where=cum.values >= 1, alpha=0.15, color=COLORS["accent"])
    ax1.fill_between(idx, 1, cum.values,
                     where=cum.values < 1, alpha=0.15, color=COLORS["secondary"])
    ax1.axhline(1, color="black", linewidth=0.8, linestyle="--")
    ax1.set_title("Cumulative Return", fontweight="bold")
    ax1.set_ylabel("Value ($)")

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(idx, 0, dd.values, color=COLORS["secondary"], alpha=0.6)
    ax2.set_title("Drawdown", fontweight="bold")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # 3. Rolling vol
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(idx, rv_21.values, color=COLORS["primary"], linewidth=1.5, label="1M")
    ax3.plot(idx, rv_63.values, color=COLORS["accent"], linewidth=1.5, label="3M")
    ax3.set_title("Rolling Volatility", fontweight="bold")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax3.legend(framealpha=0, fontsize=9)

    # 4. Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(r.values, bins=60, density=True, color=COLORS["primary"], alpha=0.5)
    x_grid = np.linspace(r.min(), r.max(), 300)
    kde = stats.gaussian_kde(r.values)
    ax4.plot(x_grid, kde(x_grid), color=COLORS["primary"], linewidth=2)
    mu_r, sigma_r = r.mean(), r.std()
    ax4.plot(x_grid, stats.norm.pdf(x_grid, mu_r, sigma_r),
             color=COLORS["accent"], linewidth=1.5, linestyle="--", label="Normal")
    var95 = float(np.quantile(r.values, 0.05))
    ax4.axvline(var95, color=COLORS["secondary"], linewidth=1.5, linestyle=":", label=f"VaR 95%={var95:.3f}")
    ax4.set_title("Return Distribution", fontweight="bold")
    ax4.legend(framealpha=0, fontsize=9)

    # 5. Key metrics table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    ann_ret = float((1 + r).prod() ** (freq / len(r)) - 1)
    ann_vol = float(r.std() * np.sqrt(freq))
    mdd = float(dd.min())
    sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else np.nan
    sortino_d = float(r[r < 0].std() * np.sqrt(freq))
    sortino = (ann_ret - risk_free) / sortino_d if sortino_d > 0 else np.nan

    table_data = [
        ["Ann. Return", f"{ann_ret:.2%}"],
        ["Ann. Volatility", f"{ann_vol:.2%}"],
        ["Sharpe Ratio", f"{sharpe:.2f}"],
        ["Sortino Ratio", f"{sortino:.2f}"],
        ["Max Drawdown", f"{mdd:.2%}"],
        ["VaR 95%", f"{var95:.4f}"],
        ["Skewness", f"{float(stats.skew(r.values)):.2f}"],
        ["Exc. Kurtosis", f"{float(stats.kurtosis(r.values, fisher=True)):.2f}"],
    ]
    tbl = ax5.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLORS["primary"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor(COLORS["light"])
    ax5.set_title("Key Metrics", fontweight="bold", pad=20)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    return _save_or_show(fig, save_path)