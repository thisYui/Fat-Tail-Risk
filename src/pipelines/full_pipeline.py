"""
pipelines/full_pipeline.py
---------------------------
Orchestrator chạy toàn bộ hệ thống end-to-end:

  FullPipeline:
    1. SimulationPipeline  – sinh dữ liệu / kịch bản
    2. ModelingPipeline    – fit phân phối, EVT, copula
    3. RiskPipeline        – metrics, stress, backtest
    4. Visualization       – tạo charts tự động
    5. Report              – xuất HTML + PDF

  run_full_analysis()    : convenience function
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .simulation_pipeline import SimulationPipeline, SimulationConfig
from .modeling_pipeline import ModelingPipeline, ModelingConfig
from .risk_pipeline import RiskPipeline, RiskConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class FullPipelineConfig:
    """
    Config tổng hợp cho toàn bộ hệ thống.

    Attributes
    ----------
    name           : tên phân tích / danh mục
    output_dir     : thư mục lưu kết quả
    run_simulation : chạy SimulationPipeline
    run_modeling   : chạy ModelingPipeline
    run_risk       : chạy RiskPipeline
    run_report     : tạo HTML/PDF report cuối
    sim_config     : SimulationConfig
    model_config   : ModelingConfig
    risk_config    : RiskConfig
    """
    name: str = "fat_tail_analysis"
    output_dir: str = "outputs"
    run_simulation: bool = True
    run_modeling: bool = True
    run_risk: bool = True
    run_report: bool = True
    # Sub-configs
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)
    model_config: ModelingConfig = field(default_factory=ModelingConfig)
    risk_config: RiskConfig = field(default_factory=RiskConfig)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class FullPipeline:
    """
    Orchestrator chạy toàn bộ Fat-Tail-Risk system end-to-end.

    Usage
    -----
    >>> cfg = FullPipelineConfig(name="SPY_analysis", output_dir="results/spy")
    >>> pipe = FullPipeline(cfg)
    >>> report = pipe.run(returns_series)
    """

    def __init__(self, config: FullPipelineConfig):
        self.config = config
        self.outputs: dict = {}
        self._setup_logging()
        self._setup_output_dir()

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    def _setup_output_dir(self) -> None:
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "figures").mkdir(exist_ok=True)
        (out / "data").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> dict:
        """
        Chạy toàn bộ pipeline end-to-end.

        Parameters
        ----------
        returns   : chuỗi return của danh mục / tài sản
        benchmark : chuỗi return benchmark (tùy chọn)

        Returns
        -------
        dict với kết quả từ tất cả các pipeline + report
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info(f"FullPipeline START: {self.config.name}")
        logger.info("=" * 60)

        r = returns if isinstance(returns, pd.Series) else pd.Series(returns)
        r = r.dropna()

        # ── Stage 1: Simulation ──────────────────────────────────────
        if self.config.run_simulation:
            self._run_simulation_stage()

        # ── Stage 2: Modeling ────────────────────────────────────────
        if self.config.run_modeling:
            self._run_modeling_stage(r.values)

        # ── Stage 3: Risk ────────────────────────────────────────────
        if self.config.run_risk:
            self._run_risk_stage(r)

        # ── Stage 4: Visualization ───────────────────────────────────
        self._run_visualization_stage(r)

        # ── Stage 5: Report ──────────────────────────────────────────
        if self.config.run_report:
            self._run_report_stage(r)

        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info(f"FullPipeline DONE in {elapsed:.1f}s")
        logger.info("=" * 60)

        return self.outputs

    # ------------------------------------------------------------------
    # Stage 1: Simulation
    # ------------------------------------------------------------------

    def _run_simulation_stage(self) -> None:
        logger.info("\n[Stage 1] Running SimulationPipeline...")
        cfg = self.config.sim_config
        cfg.output_dir = str(Path(self.config.output_dir) / "data" / "simulation")

        pipe = SimulationPipeline(cfg)
        sim_outputs = pipe.run()
        self.outputs["simulation"] = sim_outputs

        comp = sim_outputs.get("scenario_comparison")
        if comp is not None:
            logger.info(f"  Scenarios compared:\n{comp[['mean_return','var_95','cvar_95']].to_string()}")

    # ------------------------------------------------------------------
    # Stage 2: Modeling
    # ------------------------------------------------------------------

    def _run_modeling_stage(self, r: np.ndarray) -> None:
        logger.info("\n[Stage 2] Running ModelingPipeline...")
        cfg = self.config.model_config
        cfg.output_dir = str(Path(self.config.output_dir) / "data" / "modeling")

        pipe = ModelingPipeline(cfg)
        model_outputs = pipe.run(r)
        self.outputs["modeling"] = model_outputs

        best = model_outputs.get("best_distribution")
        if best:
            logger.info(f"  Best distribution: {best.dist_name} (AIC={best.aic:.2f})")

        evt = model_outputs.get("evt_summary")
        if evt is not None:
            logger.info(f"  EVT-VaR 99%: {evt.get('evt_var_99', 'N/A'):.4f}")

    # ------------------------------------------------------------------
    # Stage 3: Risk
    # ------------------------------------------------------------------

    def _run_risk_stage(self, r: pd.Series) -> None:
        logger.info("\n[Stage 3] Running RiskPipeline...")
        cfg = self.config.risk_config
        cfg.output_dir = str(Path(self.config.output_dir) / "data" / "risk")

        pipe = RiskPipeline(cfg)
        risk_outputs = pipe.run(r, name=self.config.name)
        self.outputs["risk"] = risk_outputs

        rm = risk_outputs.get("risk_metrics")
        if rm:
            logger.info(
                f"  Sharpe={rm.sharpe_ratio:.2f}  "
                f"Sortino={rm.sortino_ratio:.2f}  "
                f"MDD={rm.max_drawdown:.2%}"
            )

        bt = risk_outputs.get("backtest_report")
        if bt:
            logger.info(f"  Best VaR model: {bt.best_model()}")

    # ------------------------------------------------------------------
    # Stage 4: Visualization
    # ------------------------------------------------------------------

    def _run_visualization_stage(self, r: pd.Series) -> None:
        logger.info("\n[Stage 4] Generating visualizations...")
        fig_dir = Path(self.config.output_dir) / "figures"
        saved = []

        try:
            from ..visualization.plots import (
                plot_performance_dashboard,
                plot_return_dist,
                plot_qq,
            )
            from ..visualization.tail_plots import (
                plot_var_cvar,
                plot_hill,
                plot_mean_excess,
                plot_threshold_stability,
            )
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt

            # Dashboard
            fig = plot_performance_dashboard(r, title=f"{self.config.name} – Performance Dashboard")
            path = fig_dir / "performance_dashboard.png"
            fig.savefig(path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            saved.append(str(path))

            # Distribution
            fig = plot_return_dist(r.values, title="Return Distribution")
            path = fig_dir / "return_distribution.png"
            fig.savefig(path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            saved.append(str(path))

            # VaR/CVaR
            fig = plot_var_cvar(r.values, confidence=0.95)
            path = fig_dir / "var_cvar.png"
            fig.savefig(path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            saved.append(str(path))

            # Hill plot
            if len(r) >= 100:
                fig = plot_hill(r.values)
                path = fig_dir / "hill_plot.png"
                fig.savefig(path, bbox_inches="tight", dpi=120)
                plt.close(fig)
                saved.append(str(path))

                fig = plot_mean_excess(r.values)
                path = fig_dir / "mean_excess.png"
                fig.savefig(path, bbox_inches="tight", dpi=120)
                plt.close(fig)
                saved.append(str(path))

            logger.info(f"  Saved {len(saved)} figures to {fig_dir}")
            self.outputs["figure_paths"] = saved

        except Exception as e:
            logger.warning(f"  Visualization error: {e}")

    # ------------------------------------------------------------------
    # Stage 5: Report
    # ------------------------------------------------------------------

    def _run_report_stage(self, r: pd.Series) -> None:
        logger.info("\n[Stage 5] Generating report...")

        try:
            from ..visualization.report import generate_full_report

            html_path = str(Path(self.config.output_dir) / f"{self.config.name}_report.html")
            pdf_path = str(Path(self.config.output_dir) / f"{self.config.name}_report.pdf")

            report = generate_full_report(
                r,
                title=f"Fat-Tail Risk Report – {self.config.name}",
                author="Fat-Tail-Risk System",
                risk_free=self.config.risk_config.risk_free,
                freq=self.config.risk_config.freq,
                output_html=html_path,
                output_pdf=pdf_path,
            )
            self.outputs["report"] = report
            logger.info(f"  HTML report: {html_path}")
            logger.info(f"  PDF  report: {pdf_path}")

        except Exception as e:
            logger.warning(f"  Report generation error: {e}")

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    @property
    def risk_metrics(self):
        return self.outputs.get("risk", {}).get("risk_metrics")

    @property
    def best_distribution(self):
        return self.outputs.get("modeling", {}).get("best_distribution")

    @property
    def backtest_report(self):
        return self.outputs.get("risk", {}).get("backtest_report")

    @property
    def scenario_comparison(self):
        return self.outputs.get("simulation", {}).get("scenario_comparison")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_full_analysis(
    returns: Union[pd.Series, np.ndarray],
    name: str = "portfolio",
    output_dir: str = "outputs",
    scenarios: Optional[List[str]] = None,
    n_paths: int = 10_000,
    portfolio_value: float = 1_000_000.0,
    risk_free: float = 0.0,
    seed: int = 42,
    run_report: bool = True,
) -> FullPipeline:
    """
    Convenience function: chạy full pipeline với một lệnh.

    Parameters
    ----------
    returns        : chuỗi return đầu vào
    name           : tên phân tích
    output_dir     : thư mục output
    scenarios      : danh sách kịch bản MC (None = dùng mặc định)
    n_paths        : số MC paths
    portfolio_value: giá trị danh mục
    risk_free      : lãi suất phi rủi ro
    seed           : random seed
    run_report     : có tạo HTML/PDF report không

    Returns
    -------
    FullPipeline đã chạy xong (truy cập .outputs để lấy kết quả)

    Example
    -------
    >>> import pandas as pd
    >>> returns = pd.read_csv("spy_returns.csv", index_col=0, parse_dates=True)["return"]
    >>> pipe = run_full_analysis(returns, name="SPY", output_dir="results/spy")
    >>> print(pipe.risk_metrics.as_series())
    >>> print(pipe.scenario_comparison)
    """
    if scenarios is None:
        scenarios = ["base", "dot_com_2000", "gfc_2008", "covid_2020", "black_swan"]

    cfg = FullPipelineConfig(
        name=name,
        output_dir=output_dir,
        run_simulation=True,
        run_modeling=True,
        run_risk=True,
        run_report=run_report,
        sim_config=SimulationConfig(
            scenarios=scenarios,
            n_paths=n_paths,
            seed=seed,
        ),
        model_config=ModelingConfig(
            fit_evt=True,
            threshold_quantile=0.90,
        ),
        risk_config=RiskConfig(
            risk_free=risk_free,
            portfolio_value=portfolio_value,
            loss_threshold_for_reverse=portfolio_value * 0.10,
            run_stress=True,
            run_backtest=True,
        ),
    )

    pipe = FullPipeline(cfg)
    pipe.run(returns)
    return pipe