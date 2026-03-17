"""
pipelines/risk_pipeline.py
---------------------------
Pipeline tự động hoá đo lường rủi ro, stress test và backtest:

  RiskPipeline:
    1. Tính toàn bộ risk metrics
    2. Chạy stress test (historical + parametric + MC)
    3. Chạy VaR backtest (Kupiec + Christoffersen)
    4. Tạo rolling VaR forecasts
    5. Lưu kết quả
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..risk.metrics import compute_risk_metrics, compare_assets_risk, RiskMetrics
from ..risk.stress import (
    parametric_stress_test,
    monte_carlo_stress_summary,
    var_confidence_sensitivity,
    reverse_stress_test,
    StressReport,
)
from ..risk.backtest import (
    backtest_var_model,
    backtest_multiple_models,
    rolling_var_backtest,
    BacktestReport,
)
from ..models.var_models import hs_var, parametric_var, garch_var, filtered_hs_var

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """Cấu hình cho RiskPipeline."""
    # Risk metrics
    risk_free: float = 0.0
    freq: int = 252
    confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)
    compute_evt: bool = True

    # Stress testing
    run_stress: bool = True
    portfolio_value: float = 1_000_000.0
    mc_stress_n_simulations: int = 10_000
    mc_stress_horizon: int = 21
    loss_threshold_for_reverse: float = 100_000.0  # 10% of 1M default

    # Backtesting
    run_backtest: bool = True
    backtest_confidence: float = 0.95
    rolling_window: int = 252

    # Output
    output_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RiskPipeline:
    """
    Pipeline đo lường rủi ro toàn diện.

    Usage (single asset)
    --------------------
    >>> cfg = RiskConfig(portfolio_value=1_000_000)
    >>> pipe = RiskPipeline(cfg)
    >>> outputs = pipe.run(returns_series)
    >>> print(outputs["risk_metrics"].as_series())
    >>> print(outputs["backtest_report"].summary())

    Usage (multi-asset comparison)
    --------------------------------
    >>> outputs = pipe.run_multi({"SPY": spy_rets, "TLT": tlt_rets})
    >>> print(outputs["comparison_table"])
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.outputs: dict = {}

    def run(
        self,
        returns: Union[pd.Series, np.ndarray],
        name: str = "portfolio",
    ) -> dict:
        """
        Chạy pipeline cho một chuỗi return.

        Parameters
        ----------
        returns : chuỗi return (pd.Series với DatetimeIndex hoặc array)
        name    : tên danh mục

        Returns
        -------
        dict với keys:
            risk_metrics, var_series, cvar_series,
            stress_report, mc_stress_summary, reverse_stress,
            backtest_report, rolling_var_df
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns, name=name)

        r = returns.dropna()
        logger.info(f"RiskPipeline: {name} | n={len(r)}")

        # 1. Risk metrics
        self._compute_metrics(r.values)

        # 2. Rolling VaR series
        self._compute_rolling_var(r)

        # 3. Stress testing
        if self.config.run_stress:
            self._run_stress_tests(r.values)

        # 4. Backtesting
        if self.config.run_backtest:
            self._run_backtest(r)

        # 5. Save
        if self.config.output_dir:
            self._save_outputs(name)

        logger.info("RiskPipeline completed.")
        return self.outputs

    def run_multi(
        self,
        returns_dict: Dict[str, Union[pd.Series, np.ndarray]],
    ) -> dict:
        """
        Chạy pipeline cho nhiều tài sản / chiến lược đồng thời.

        Parameters
        ----------
        returns_dict : dict {name: returns}

        Returns
        -------
        dict với key 'comparison_table' + kết quả từng tài sản
        """
        for name, rets in returns_dict.items():
            logger.info(f"Processing {name}...")
            self.run(rets, name=name)

        # Comparison table
        arr_dict = {
            k: (v.values if isinstance(v, pd.Series) else v)
            for k, v in returns_dict.items()
        }
        comp = compare_assets_risk(
            arr_dict,
            risk_free=self.config.risk_free,
            freq=self.config.freq,
        )
        self.outputs["comparison_table"] = comp
        logger.info("Multi-asset comparison complete.")
        return self.outputs

    # ------------------------------------------------------------------
    # Step 1: Risk metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, r: np.ndarray) -> None:
        logger.info("  Computing risk metrics...")
        rm = compute_risk_metrics(
            r,
            risk_free=self.config.risk_free,
            freq=self.config.freq,
            compute_evt=self.config.compute_evt,
        )
        self.outputs["risk_metrics"] = rm
        logger.info(
            f"  Sharpe={rm.sharpe_ratio:.2f}, VaR95={rm.var_95:.4f}, "
            f"CVaR95={rm.cvar_95:.4f}, MDD={rm.max_drawdown:.2%}"
        )

    # ------------------------------------------------------------------
    # Step 2: Rolling VaR
    # ------------------------------------------------------------------

    def _compute_rolling_var(self, r: pd.Series) -> None:
        logger.info("  Computing rolling VaR...")
        alpha = 1 - self.config.backtest_confidence
        window = self.config.rolling_window

        rolling_var_95 = r.rolling(window).quantile(1 - 0.95)
        rolling_var_99 = r.rolling(window).quantile(1 - 0.99)
        rolling_vol = r.rolling(window).std() * np.sqrt(self.config.freq)

        var_df = pd.DataFrame({
            "return": r,
            "rolling_var_95": rolling_var_95,
            "rolling_var_99": rolling_var_99,
            "rolling_vol": rolling_vol,
        })
        self.outputs["rolling_var_df"] = var_df

    # ------------------------------------------------------------------
    # Step 3: Stress tests
    # ------------------------------------------------------------------

    def _run_stress_tests(self, r: np.ndarray) -> None:
        logger.info("  Running stress tests...")
        cfg = self.config

        # Parametric scenarios
        try:
            stress_report = parametric_stress_test(
                r,
                portfolio_value=cfg.portfolio_value,
                confidence=cfg.backtest_confidence,
            )
            self.outputs["stress_report"] = stress_report
            worst = stress_report.worst_scenarios(5)
            logger.info(f"  Worst 5 scenarios:\n{worst[['pnl', 'return_estimate']].to_string()}")
        except Exception as e:
            logger.warning(f"  Parametric stress failed: {e}")

        # Monte Carlo stress
        try:
            mc_stress = monte_carlo_stress_summary(
                r,
                n_simulations=cfg.mc_stress_n_simulations,
                horizon=cfg.mc_stress_horizon,
                confidence_levels=cfg.confidence_levels,
            )
            self.outputs["mc_stress_summary"] = mc_stress
        except Exception as e:
            logger.warning(f"  MC stress failed: {e}")

        # VaR sensitivity
        try:
            var_sens = var_confidence_sensitivity(r)
            self.outputs["var_sensitivity"] = var_sens
        except Exception as e:
            logger.warning(f"  VaR sensitivity failed: {e}")

        # Reverse stress test
        try:
            rev = reverse_stress_test(
                r,
                portfolio_value=cfg.portfolio_value,
                loss_threshold=cfg.loss_threshold_for_reverse,
            )
            self.outputs["reverse_stress"] = rev
            logger.info(
                f"  Reverse stress: required_return={rev['required_return']:.4f}, "
                f"frequency={rev['historical_frequency']:.4f}"
            )
        except Exception as e:
            logger.warning(f"  Reverse stress failed: {e}")

    # ------------------------------------------------------------------
    # Step 4: Backtest
    # ------------------------------------------------------------------

    def _run_backtest(self, r: pd.Series) -> None:
        logger.info("  Running VaR backtests...")
        cfg = self.config
        r_arr = r.values
        n = len(r_arr)
        window = cfg.rolling_window
        cl = cfg.backtest_confidence
        alpha = 1 - cl

        if n < window + 10:
            logger.warning(f"  Too few observations for backtest (n={n} < window+10)")
            return

        # Build rolling VaR forecasts for each method
        model_forecasts: Dict[str, np.ndarray] = {}
        methods = {
            "hs": lambda arr: hs_var(arr, cl).var,
            "parametric_normal": lambda arr: parametric_var(arr, cl, "normal").var,
            "parametric_t": lambda arr: parametric_var(arr, cl, "student_t").var,
            "garch_normal": lambda arr: garch_var(arr, cl, dist="normal").var,
            "filtered_hs": lambda arr: filtered_hs_var(arr, cl).var,
        }

        for method_name, fn in methods.items():
            try:
                forecasts = np.full(n, np.nan)
                for t in range(window, n):
                    forecasts[t] = fn(r_arr[t - window: t])
                model_forecasts[method_name] = forecasts
            except Exception as e:
                logger.warning(f"  Rolling forecast failed for {method_name}: {e}")

        # Trim to valid range
        valid_start = window
        r_valid = r_arr[valid_start:]
        model_forecasts_valid = {
            k: v[valid_start:] for k, v in model_forecasts.items()
        }

        # Run backtests
        try:
            backtest_report = backtest_multiple_models(
                r_valid, model_forecasts_valid, confidence=cl
            )
            self.outputs["backtest_report"] = backtest_report
            summary = backtest_report.summary()
            logger.info(f"  Best model: {backtest_report.best_model()}")
            logger.info(f"\n{summary[['n_violations','violation_rate','kupiec_pvalue','traffic_light']].to_string()}")
        except Exception as e:
            logger.warning(f"  Backtest failed: {e}")

        # Rolling backtest DF (HS only for visualization)
        if "hs" in model_forecasts:
            try:
                rolling_df = pd.DataFrame({
                    "date": r.index[valid_start:],
                    "return": r_valid,
                    "var_forecast": model_forecasts_valid["hs"],
                    "violation": r_valid < model_forecasts_valid["hs"],
                }).set_index("date")
                rolling_df["rolling_viol_rate"] = (
                    rolling_df["violation"].rolling(window).mean()
                )
                self.outputs["rolling_backtest_df"] = rolling_df
            except Exception as e:
                logger.warning(f"  Rolling backtest DF failed: {e}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_outputs(self, name: str) -> None:
        out_dir = Path(self.config.output_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Risk metrics
        rm: Optional[RiskMetrics] = self.outputs.get("risk_metrics")
        if rm:
            rm.as_series().to_frame("value").to_csv(out_dir / "risk_metrics.csv")

        # Rolling VaR
        rvdf: Optional[pd.DataFrame] = self.outputs.get("rolling_var_df")
        if rvdf is not None:
            rvdf.to_csv(out_dir / "rolling_var.csv")

        # VaR sensitivity
        sens: Optional[pd.DataFrame] = self.outputs.get("var_sensitivity")
        if sens is not None:
            sens.to_csv(out_dir / "var_sensitivity.csv")

        # MC stress summary
        mc_stress: Optional[pd.Series] = self.outputs.get("mc_stress_summary")
        if mc_stress is not None:
            mc_stress.to_frame("value").to_csv(out_dir / "mc_stress_summary.csv")

        # Backtest report
        bt: Optional[BacktestReport] = self.outputs.get("backtest_report")
        if bt:
            bt.summary().to_csv(out_dir / "backtest_summary.csv")

        logger.info(f"  Outputs saved to {out_dir}")