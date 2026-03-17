"""
src/__init__.py
================
Top-level public API của Fat-Tail-Risk package.

Cấu trúc subpackages
---------------------
  src.features      – tính toán return, moments, tail features
  src.simulation    – sinh dữ liệu (generators, processes, scenarios, MC)
  src.models        – phân phối, VaR, CVaR, EVT, Copula
  src.risk          – metrics, stress testing, backtesting
  src.visualization – plots, tail plots, report builder
  src.pipelines     – simulation / modeling / risk / full pipeline

Quick start
-----------
>>> from src import run_full_analysis
>>> pipe = run_full_analysis(returns, name="SPY", output_dir="results/spy")

>>> from src.features import log_returns, tail_summary
>>> from src.models import hs_var, evt_var, fit_all
>>> from src.risk import compute_risk_metrics, backtest_multiple_models
>>> from src.simulation import MonteCarloEngine, get_scenario
>>> from src.visualization import plot_performance_dashboard, generate_full_report
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("fat-tail-risk")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "Fat-Tail-Risk"
__description__ = "Fat-tail risk modelling: EVT, Copula, VaR/CVaR, stress testing, backtesting"

# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
from .features import (
    # returns
    simple_returns,
    log_returns,
    excess_returns,
    rolling_returns,
    annualised_return,
    cumulative_returns,
    drawdown,
    max_drawdown,
    return_summary,
    # moments
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
    # tail
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

# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
from .simulation import (
    # generators
    normal,
    multivariate_normal,
    student_t,
    multivariate_student_t,
    skewed_student_t,
    alpha_stable,
    generalized_pareto,
    laplace,
    mixture_of_normals,
    sample,
    DISTRIBUTION_MAP,
    # processes
    gbm,
    gbm_returns,
    merton_jump_diffusion,
    garch_11,
    egarch_11,
    heston,
    ornstein_uhlenbeck,
    fractional_brownian_motion,
    # scenarios
    ScenarioConfig,
    FactorShock,
    CRISIS_SCENARIOS,
    FACTOR_SHOCKS,
    get_scenario,
    list_scenarios,
    perturb_scenario,
    scenario_grid,
    apply_factor_shock,
    bootstrap_scenarios,
    worst_scenarios,
    # monte carlo
    MCResult,
    MonteCarloEngine,
    run_mc_prices,
    run_mc_returns,
    portfolio_mc,
    run_scenario_comparison,
    compare_scenarios_summary,
    summarise_mc,
)

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
from .models import (
    # distribution
    DistributionFit,
    fit_normal,
    fit_student_t,
    fit_skewed_t,
    fit_nig,
    fit_gpd,
    fit_all,
    best_fit,
    fitted_quantile,
    fitted_pdf,
    FITTERS,
    # VaR
    VaRResult,
    hs_var,
    rolling_hs_var,
    parametric_var,
    cornish_fisher_var,
    garch_var,
    filtered_hs_var,
    compare_var_methods,
    # CVaR
    CVaRResult,
    historical_cvar,
    rolling_historical_cvar,
    parametric_cvar,
    garch_cvar,
    filtered_hs_cvar,
    cvar_contribution,
    compare_cvar_methods,
    # EVT
    GEVResult,
    GPDResult,
    fit_gev,
    fit_gpd_pot,
    evt_var,
    evt_cvar,
    return_levels,
    threshold_stability_plot_data,
    mean_excess_plot_data,
    evt_summary,
    # Copula
    CopulaFamily,
    CopulaResult,
    pseudo_observations,
    fit_gaussian_copula,
    sample_gaussian_copula,
    fit_student_t_copula,
    sample_student_t_copula,
    fit_archimedean_copula,
    sample_archimedean_copula,
    fit_all_copulas,
    empirical_tail_dependence,
    tail_dependence_matrix,
)

# ---------------------------------------------------------------------------
# risk
# ---------------------------------------------------------------------------
from .risk import (
    # metrics
    RiskMetrics,
    annualised_vol,
    downside_deviation,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    omega_ratio,
    information_ratio,
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
    # stress
    StressResult,
    StressReport,
    historical_stress_test,
    parametric_stress_test,
    factor_stress_test,
    sensitivity_analysis,
    var_confidence_sensitivity,
    reverse_stress_test,
    monte_carlo_stress_summary,
    # backtest
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

# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------
from .visualization import (
    # general plots
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
    # tail plots
    plot_var_cvar,
    plot_evt_tail,
    plot_hill,
    plot_mean_excess,
    plot_threshold_stability,
    plot_backtest_violations,
    plot_scenario_fan,
    plot_tail_dependence,
    plot_scenario_comparison,
    # report
    ReportSection,
    RiskReport,
    generate_full_report,
)

# ---------------------------------------------------------------------------
# pipelines
# ---------------------------------------------------------------------------
from .pipelines import (
    SimulationPipeline,
    SimulationConfig,
    ModelingPipeline,
    ModelingConfig,
    RiskPipeline,
    RiskConfig,
    FullPipeline,
    FullPipelineConfig,
    run_full_analysis,
)

# ---------------------------------------------------------------------------
# Resolve name conflicts between subpackages
# ---------------------------------------------------------------------------
# features.annualised_return  vs  risk.annualised_return (same logic, alias)
# features.max_drawdown       vs  risk.max_drawdown       (same logic, alias)
# features.tail_ratio         vs  features.tail.tail_ratio (same)
# → keep both accessible; top-level names point to risk.* versions for
#   portfolio-level usage, features.* still accessible via subpackage.

from .risk.metrics import (
    annualised_return as annualised_return,
    max_drawdown as max_drawdown,
    drawdown_series,
)

# ---------------------------------------------------------------------------
# __all__: explicit public surface
# ---------------------------------------------------------------------------
__all__ = [
    # meta
    "__version__", "__author__", "__description__",

    # ── features ──────────────────────────────────────────────────────
    # returns
    "simple_returns", "log_returns", "excess_returns", "rolling_returns",
    "annualised_return", "cumulative_returns", "drawdown", "max_drawdown",
    "return_summary",
    # moments
    "mean", "variance", "std", "skewness", "kurtosis", "semi_deviation",
    "rolling_mean", "rolling_std", "rolling_skewness", "rolling_kurtosis",
    "jarque_bera_test", "shapiro_wilk_test", "moments_summary",
    # tail features
    "empirical_var", "empirical_cvar", "hill_estimator", "hill_plot",
    "tail_ratio", "mean_excess_function", "select_pot_threshold",
    "pot_exceedances", "tail_probability", "tail_quantile", "tail_summary",

    # ── simulation ────────────────────────────────────────────────────
    "normal", "multivariate_normal", "student_t", "multivariate_student_t",
    "skewed_student_t", "alpha_stable", "generalized_pareto", "laplace",
    "mixture_of_normals", "sample", "DISTRIBUTION_MAP",
    "gbm", "gbm_returns", "merton_jump_diffusion", "garch_11", "egarch_11",
    "heston", "ornstein_uhlenbeck", "fractional_brownian_motion",
    "ScenarioConfig", "FactorShock", "CRISIS_SCENARIOS", "FACTOR_SHOCKS",
    "get_scenario", "list_scenarios", "perturb_scenario", "scenario_grid",
    "apply_factor_shock", "bootstrap_scenarios", "worst_scenarios",
    "MCResult", "MonteCarloEngine", "run_mc_prices", "run_mc_returns",
    "portfolio_mc", "run_scenario_comparison", "compare_scenarios_summary",
    "summarise_mc",

    # ── models ────────────────────────────────────────────────────────
    "DistributionFit", "fit_normal", "fit_student_t", "fit_skewed_t",
    "fit_nig", "fit_gpd", "fit_all", "best_fit", "fitted_quantile",
    "fitted_pdf", "FITTERS",
    "VaRResult", "hs_var", "rolling_hs_var", "parametric_var",
    "cornish_fisher_var", "garch_var", "filtered_hs_var", "compare_var_methods",
    "CVaRResult", "historical_cvar", "rolling_historical_cvar",
    "parametric_cvar", "garch_cvar", "filtered_hs_cvar",
    "cvar_contribution", "compare_cvar_methods",
    "GEVResult", "GPDResult", "fit_gev", "fit_gpd_pot",
    "evt_var", "evt_cvar", "return_levels",
    "threshold_stability_plot_data", "mean_excess_plot_data", "evt_summary",
    "CopulaFamily", "CopulaResult", "pseudo_observations",
    "fit_gaussian_copula", "sample_gaussian_copula",
    "fit_student_t_copula", "sample_student_t_copula",
    "fit_archimedean_copula", "sample_archimedean_copula",
    "fit_all_copulas", "empirical_tail_dependence", "tail_dependence_matrix",

    # ── risk ──────────────────────────────────────────────────────────
    "RiskMetrics", "annualised_vol", "downside_deviation",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio",
    "information_ratio", "gain_to_pain_ratio",
    "expected_tail_loss", "entropic_var", "tail_gini",
    "marginal_var", "component_var", "percentage_component_var",
    "risk_parity_weights", "compute_risk_metrics", "compare_assets_risk",
    "drawdown_series",
    "StressResult", "StressReport",
    "historical_stress_test", "parametric_stress_test",
    "factor_stress_test", "sensitivity_analysis",
    "var_confidence_sensitivity", "reverse_stress_test",
    "monte_carlo_stress_summary",
    "BacktestResult", "BacktestReport",
    "compute_violations", "kupiec_test", "christoffersen_test",
    "conditional_coverage_test", "basel_traffic_light",
    "mcneil_frey_test", "es_bootstrap_test",
    "rolling_var_backtest", "backtest_var_model",
    "backtest_multiple_models", "binomial_var_test",

    # ── visualization ─────────────────────────────────────────────────
    "plot_returns", "plot_cumulative", "plot_drawdown", "plot_rolling_vol",
    "plot_return_dist", "plot_qq", "plot_acf_pacf",
    "plot_risk_comparison", "plot_correlation",
    "plot_performance_dashboard", "COLORS",
    "plot_var_cvar", "plot_evt_tail", "plot_hill", "plot_mean_excess",
    "plot_threshold_stability", "plot_backtest_violations",
    "plot_scenario_fan", "plot_tail_dependence", "plot_scenario_comparison",
    "ReportSection", "RiskReport", "generate_full_report",

    # ── pipelines ─────────────────────────────────────────────────────
    "SimulationPipeline", "SimulationConfig",
    "ModelingPipeline", "ModelingConfig",
    "RiskPipeline", "RiskConfig",
    "FullPipeline", "FullPipelineConfig",
    "run_full_analysis",
]