"""
simulation submodule
--------------------
Monte Carlo and stochastic process simulation.
"""

from __future__ import annotations

from .monte_carlo import (
    simulate_from_distribution,
    simulate_quantiles,
    monte_carlo_tail_probability,
    simulate_from_empirical,
)
from .stochastic_processes import (
    geometric_brownian_motion,
    gbm_with_t_innovations,
    ornstein_uhlenbeck,
    jump_diffusion,
)

__all__ = [
    "simulate_from_distribution",
    "simulate_quantiles",
    "monte_carlo_tail_probability",
    "simulate_from_empirical",
    "geometric_brownian_motion",
    "gbm_with_t_innovations",
    "ornstein_uhlenbeck",
    "jump_diffusion",
]
