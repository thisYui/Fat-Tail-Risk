"""
src/models/__init__.py
-----------------------
Public API của models package.
"""

# Distribution fitting
from .distribution import (
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
)

# VaR models
from .var_models import (
    VaRResult,
    hs_var,
    rolling_hs_var,
    parametric_var,
    cornish_fisher_var,
    garch_var,
    filtered_hs_var,
    compare_var_methods,
)

# CVaR / Expected Shortfall
from .cvar_models import (
    CVaRResult,
    historical_cvar,
    rolling_historical_cvar,
    parametric_cvar,
    garch_cvar,
    filtered_hs_cvar,
    cvar_contribution,
    compare_cvar_methods,
)

# Extreme Value Theory
from .evt import (
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
)

# Copula models
from .copula import (
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

__all__ = [
    # distribution
    "DistributionFit", "fit_normal", "fit_student_t", "fit_skewed_t",
    "fit_nig", "fit_gpd", "fit_all", "best_fit", "fitted_quantile",
    "fitted_pdf", "FITTERS",
    # var
    "VaRResult", "hs_var", "rolling_hs_var", "parametric_var",
    "cornish_fisher_var", "garch_var", "filtered_hs_var", "compare_var_methods",
    # cvar
    "CVaRResult", "historical_cvar", "rolling_historical_cvar",
    "parametric_cvar", "garch_cvar", "filtered_hs_cvar",
    "cvar_contribution", "compare_cvar_methods",
    # evt
    "GEVResult", "GPDResult", "fit_gev", "fit_gpd_pot",
    "evt_var", "evt_cvar", "return_levels",
    "threshold_stability_plot_data", "mean_excess_plot_data", "evt_summary",
    # copula
    "CopulaFamily", "CopulaResult", "pseudo_observations",
    "fit_gaussian_copula", "sample_gaussian_copula",
    "fit_student_t_copula", "sample_student_t_copula",
    "fit_archimedean_copula", "sample_archimedean_copula",
    "fit_all_copulas", "empirical_tail_dependence", "tail_dependence_matrix",
]