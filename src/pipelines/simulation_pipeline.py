"""
pipelines/simulation_pipeline.py
----------------------------------
Pipeline tự động hoá quá trình sinh dữ liệu mô phỏng:

  SimulationPipeline:
    1. Nhận ScenarioConfig hoặc tên kịch bản
    2. Chạy Monte Carlo (run_mc_prices)
    3. Tính summary statistics
    4. (Tùy chọn) Sinh portfolio MC với nhiều tài sản
    5. Lưu kết quả ra file
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..simulation.monte_carlo import (
    MCResult,
    MonteCarloEngine,
    run_mc_prices,
    portfolio_mc,
    compare_scenarios_summary,
    summarise_mc,
)
from ..simulation.scenarios import (
    ScenarioConfig,
    CRISIS_SCENARIOS,
    list_scenarios,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Cấu hình cho SimulationPipeline."""
    scenarios: List[str] = field(default_factory=lambda: ["base", "gfc_2008", "covid_2020"])
    n_paths: int = 10_000
    n_steps: int = 252
    S0: float = 100.0
    dt: float = 1 / 252
    seed: Optional[int] = 42
    confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)
    # Portfolio settings (optional)
    run_portfolio: bool = False
    portfolio_weights: Optional[List[float]] = None
    portfolio_mu: Optional[List[float]] = None
    portfolio_cov: Optional[List[List[float]]] = None
    portfolio_dist: str = "normal"
    # Output
    output_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SimulationPipeline:
    """
    Pipeline chạy Monte Carlo simulation cho nhiều kịch bản.

    Usage
    -----
    >>> cfg = SimulationConfig(scenarios=["base", "gfc_2008"], n_paths=50_000)
    >>> pipe = SimulationPipeline(cfg)
    >>> results = pipe.run()
    >>> print(results["scenario_comparison"])
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.engine = MonteCarloEngine(
            n_paths=config.n_paths,
            n_steps=config.n_steps,
            S0=config.S0,
            dt=config.dt,
            seed=config.seed,
        )
        self.results: Dict[str, MCResult] = {}
        self.outputs: dict = {}

    def run(self) -> dict:
        """
        Chạy toàn bộ pipeline.

        Returns
        -------
        dict với keys:
            mc_results           : Dict[str, MCResult]
            scenario_comparison  : pd.DataFrame
            mc_summaries         : Dict[str, pd.Series]
            portfolio_result     : MCResult (nếu run_portfolio=True)
        """
        logger.info("Starting SimulationPipeline")

        # 1. Run MC cho từng kịch bản
        self._run_scenarios()

        # 2. So sánh kịch bản
        self._compare_scenarios()

        # 3. Portfolio MC (tùy chọn)
        if self.config.run_portfolio:
            self._run_portfolio_mc()

        # 4. Lưu kết quả
        if self.config.output_dir:
            self._save_outputs()

        logger.info("SimulationPipeline completed")
        return self.outputs

    def _run_scenarios(self) -> None:
        logger.info(f"Running MC for {len(self.config.scenarios)} scenarios...")
        summaries = {}
        for name in self.config.scenarios:
            try:
                result = self.engine.run(name)
                self.results[name] = result
                summaries[name] = summarise_mc(
                    result.returns,
                    confidence_levels=self.config.confidence_levels,
                )
                logger.info(
                    f"  {name}: mean={result.returns.mean():.4f}, "
                    f"VaR95={result.var(0.95):.4f}"
                )
            except Exception as e:
                logger.warning(f"  Failed {name}: {e}")

        self.outputs["mc_results"] = self.results
        self.outputs["mc_summaries"] = summaries

    def _compare_scenarios(self) -> None:
        if not self.results:
            return
        comp = compare_scenarios_summary(
            self.results,
            confidence_levels=self.config.confidence_levels,
        )
        self.outputs["scenario_comparison"] = comp
        logger.info("Scenario comparison table built.")

    def _run_portfolio_mc(self) -> None:
        cfg = self.config
        if cfg.portfolio_weights is None or cfg.portfolio_mu is None or cfg.portfolio_cov is None:
            logger.warning("Portfolio MC skipped: missing weights / mu / cov")
            return

        weights = np.array(cfg.portfolio_weights)
        mu_vec = np.array(cfg.portfolio_mu)
        cov_matrix = np.array(cfg.portfolio_cov)

        result = portfolio_mc(
            weights=weights,
            mu_vec=mu_vec,
            cov_matrix=cov_matrix,
            n_paths=cfg.n_paths,
            n_steps=cfg.n_steps,
            dist=cfg.portfolio_dist,
            seed=cfg.seed,
        )
        self.outputs["portfolio_result"] = result
        self.outputs["portfolio_summary"] = summarise_mc(
            result.returns,
            confidence_levels=cfg.confidence_levels,
        )
        logger.info(
            f"Portfolio MC done: VaR95={result.var(0.95):.4f}, CVaR95={result.cvar(0.95):.4f}"
        )

    def _save_outputs(self) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Scenario comparison CSV
        if "scenario_comparison" in self.outputs:
            path = out_dir / "scenario_comparison.csv"
            self.outputs["scenario_comparison"].to_csv(path)
            logger.info(f"Saved: {path}")

        # MC summaries CSV
        if "mc_summaries" in self.outputs:
            summary_df = pd.DataFrame(self.outputs["mc_summaries"])
            path = out_dir / "mc_summaries.csv"
            summary_df.to_csv(path)
            logger.info(f"Saved: {path}")

        # Portfolio summary
        if "portfolio_summary" in self.outputs:
            path = out_dir / "portfolio_summary.csv"
            self.outputs["portfolio_summary"].to_frame("value").to_csv(path)
            logger.info(f"Saved: {path}")

    def get_scenario_returns(self, scenario: str) -> np.ndarray:
        """Trả về mảng total returns của một kịch bản."""
        if scenario not in self.results:
            raise KeyError(f"Scenario '{scenario}' not run yet.")
        return self.results[scenario].returns

    def get_worst_paths(self, scenario: str, n: int = 10) -> np.ndarray:
        """Trả về n đường (path) tệ nhất của một kịch bản."""
        result = self.results[scenario]
        worst_idx = np.argsort(result.returns)[:n]
        return result.paths[:, worst_idx]