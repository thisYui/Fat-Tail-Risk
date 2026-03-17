"""
src/features/__init__.py
------------------------
Public API của features package.
"""

from .returns import (
    simple_returns,
    log_returns,
    excess_returns,
    rolling_returns,
    annualised_return,
    cumulative_returns,
    drawdown,
    max_drawdown,
    return_summary,
)

from .moments import (
    mean,
    variance,
    std,
    skewness,
    kurtosis,
    semi_deviation,
    rolling_mean,
    rolling_std,
    rolling_skewness,
    rolling_kurtosis,
    jarque_bera_test,
    shapiro_wilk_test,
    moments_summary,
)

from .tail import (
    empirical_var,
    empirical_cvar,
    hill_estimator,
    hill_plot,
    tail_ratio,
    mean_excess_function,
    select_pot_threshold,
    pot_exceedances,
    tail_probability,
    tail_quantile,
    tail_summary,
)

__all__ = [
    # returns
    "simple_returns", "log_returns", "excess_returns", "rolling_returns",
    "annualised_return", "cumulative_returns", "drawdown", "max_drawdown",
    "return_summary",
    # moments
    "mean", "variance", "std", "skewness", "kurtosis", "semi_deviation",
    "rolling_mean", "rolling_std", "rolling_skewness", "rolling_kurtosis",
    "jarque_bera_test", "shapiro_wilk_test", "moments_summary",
    # tail
    "empirical_var", "empirical_cvar", "hill_estimator", "hill_plot",
    "tail_ratio", "mean_excess_function", "select_pot_threshold",
    "pot_exceedances", "tail_probability", "tail_quantile", "tail_summary",
]