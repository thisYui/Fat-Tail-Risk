"""
src/visualization/__init__.py
------------------------------
Public API của visualization package.
"""

from .plots import (
    plot_returns,
    plot_cumulative,
    plot_drawdown,
    plot_rolling_vol,
    plot_return_dist,
    plot_qq,
    plot_acf_pacf,
    plot_risk_comparison,
    plot_correlation,
    plot_performance_dashboard,
    COLORS,
)

from .tail_plots import (
    plot_var_cvar,
    plot_evt_tail,
    plot_hill,
    plot_mean_excess,
    plot_threshold_stability,
    plot_backtest_violations,
    plot_scenario_fan,
    plot_tail_dependence,
    plot_scenario_comparison,
)

from .report import (
    ReportSection,
    RiskReport,
    generate_full_report,
)

__all__ = [
    # plots
    "plot_returns", "plot_cumulative", "plot_drawdown", "plot_rolling_vol",
    "plot_return_dist", "plot_qq", "plot_acf_pacf", "plot_risk_comparison",
    "plot_correlation", "plot_performance_dashboard", "COLORS",
    # tail plots
    "plot_var_cvar", "plot_evt_tail", "plot_hill", "plot_mean_excess",
    "plot_threshold_stability", "plot_backtest_violations",
    "plot_scenario_fan", "plot_tail_dependence", "plot_scenario_comparison",
    # report
    "ReportSection", "RiskReport", "generate_full_report",
]