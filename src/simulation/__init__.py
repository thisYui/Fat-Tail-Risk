"""
src/simulation/__init__.py
--------------------------
Public API của simulation package.
"""

# Generators – primitive distributions
from .generators import (
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
)

# Processes – stochastic price/return processes
from .processes import (
    gbm,
    gbm_returns,
    merton_jump_diffusion,
    garch_11,
    egarch_11,
    heston,
    ornstein_uhlenbeck,
    fractional_brownian_motion,
)

# Scenarios
from .scenarios import (
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
)

# Monte Carlo engine
from .monte_carlo import (
    MCResult,
    MonteCarloEngine,
    run_mc_prices,
    run_mc_returns,
    portfolio_mc,
    run_scenario_comparison,
    compare_scenarios_summary,
    summarise_mc,
)

__all__ = [
    # generators
    "normal", "multivariate_normal", "student_t", "multivariate_student_t",
    "skewed_student_t", "alpha_stable", "generalized_pareto", "laplace",
    "mixture_of_normals", "sample", "DISTRIBUTION_MAP",
    # processes
    "gbm", "gbm_returns", "merton_jump_diffusion", "garch_11", "egarch_11",
    "heston", "ornstein_uhlenbeck", "fractional_brownian_motion",
    # scenarios
    "ScenarioConfig", "FactorShock", "CRISIS_SCENARIOS", "FACTOR_SHOCKS",
    "get_scenario", "list_scenarios", "perturb_scenario", "scenario_grid",
    "apply_factor_shock", "bootstrap_scenarios", "worst_scenarios",
    # monte carlo
    "MCResult", "MonteCarloEngine", "run_mc_prices", "run_mc_returns",
    "portfolio_mc", "run_scenario_comparison", "compare_scenarios_summary",
    "summarise_mc",
]