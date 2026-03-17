"""
src/risk/__init__.py
---------------------
Public API của risk package.
"""

# Risk metrics
from .metrics import (
    RiskMetrics,
    annualised_return,
    annualised_vol,
    downside_deviation,
    max_drawdown,
    drawdown_series,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    omega_ratio,
    information_ratio,
    tail_ratio,
    gain_to_pain_ratio,
    expected_tail_loss,
    entropic_var,
    tail_gini,
    marginal_var,
    component_var,
    percentage_component_var,
    risk_parity_weights,
    compute_risk_metrics,
    compare_assets_risk,
)

# Stress testing
from .stress import (
    StressResult,
    StressReport,
    historical_stress_test,
    parametric_stress_test,
    factor_stress_test,
    sensitivity_analysis,
    var_confidence_sensitivity,
    reverse_stress_test,
    monte_carlo_stress_summary,
)

# Backtesting
from .backtest import (
    BacktestResult,
    BacktestReport,
    compute_violations,
    kupiec_test,
    christoffersen_test,
    conditional_coverage_test,
    basel_traffic_light,
    mcneil_frey_test,
    es_bootstrap_test,
    rolling_var_backtest,
    backtest_var_model,
    backtest_multiple_models,
    binomial_var_test,
)

__all__ = [
    # metrics
    "RiskMetrics", "annualised_return", "annualised_vol",
    "downside_deviation", "max_drawdown", "drawdown_series",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio",
    "information_ratio", "tail_ratio", "gain_to_pain_ratio",
    "expected_tail_loss", "entropic_var", "tail_gini",
    "marginal_var", "component_var", "percentage_component_var",
    "risk_parity_weights", "compute_risk_metrics", "compare_assets_risk",
    # stress
    "StressResult", "StressReport",
    "historical_stress_test", "parametric_stress_test",
    "factor_stress_test", "sensitivity_analysis",
    "var_confidence_sensitivity", "reverse_stress_test",
    "monte_carlo_stress_summary",
    # backtest
    "BacktestResult", "BacktestReport",
    "compute_violations", "kupiec_test", "christoffersen_test",
    "conditional_coverage_test", "basel_traffic_light",
    "mcneil_frey_test", "es_bootstrap_test",
    "rolling_var_backtest", "backtest_var_model",
    "backtest_multiple_models", "binomial_var_test",
]