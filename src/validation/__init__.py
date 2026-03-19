"""
validation submodule
--------------------
Statistical validation and goodness-of-fit testing.
"""

from __future__ import annotations

from .goodness_of_fit import (
    kolmogorov_smirnov_test,
    anderson_darling_test,
    cramer_von_mises_test,
    goodness_of_fit_summary,
)
from .qq import (
    qq_plot,
    pp_plot,
    multi_qq_plot,
)
from .statistical_tests import (
    likelihood_ratio_test,
    vuong_test,
    jarque_bera_test,
    dagostino_pearson_test,
    shapiro_wilk_test,
)

__all__ = [
    "kolmogorov_smirnov_test",
    "anderson_darling_test",
    "cramer_von_mises_test",
    "goodness_of_fit_summary",
    "qq_plot",
    "pp_plot",
    "multi_qq_plot",
    "likelihood_ratio_test",
    "vuong_test",
    "jarque_bera_test",
    "dagostino_pearson_test",
    "shapiro_wilk_test",
]
