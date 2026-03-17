"""
pipelines/modeling_pipeline.py
--------------------------------
Pipeline tự động hoá quá trình fit mô hình thống kê:

  ModelingPipeline:
    1. Fit các phân phối (Normal, t, NIG, Skewed-t)
    2. Fit GPD (EVT/POT)
    3. Fit Copula (Gaussian, t, Archimedean)
    4. So sánh model AIC/BIC
    5. Tính VaR/CVaR từ model tốt nhất
    6. Lưu kết quả
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..models.distribution import fit_all, best_fit, DistributionFit
from ..models.evt import fit_gpd_pot, evt_summary, GPDResult
from ..models.var_models import compare_var_methods, VaRResult
from ..models.cvar_models import compare_cvar_methods, CVaRResult
from ..models.copula import (
    pseudo_observations,
    fit_all_copulas,
    CopulaResult,
    empirical_tail_dependence,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelingConfig:
    """Cấu hình cho ModelingPipeline."""
    distributions: List[str] = field(
        default_factory=lambda: ["normal", "student_t", "skewed_t", "nig"]
    )
    fit_evt: bool = True
    threshold_quantile: float = 0.90
    fit_copula: bool = False             # chỉ áp dụng nếu có nhiều tài sản
    copula_families: List[str] = field(
        default_factory=lambda: ["gaussian", "student_t", "clayton", "gumbel"]
    )
    confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)
    rank_by: str = "aic"
    output_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ModelingPipeline:
    """
    Pipeline fit và đánh giá mô hình phân phối + EVT + Copula.

    Usage (đơn tài sản)
    --------------------
    >>> cfg = ModelingConfig()
    >>> pipe = ModelingPipeline(cfg)
    >>> outputs = pipe.run(returns_array)
    >>> print(outputs["best_distribution"])
    >>> print(outputs["evt_summary"])
    >>> print(outputs["var_comparison"])

    Usage (đa tài sản – copula)
    ----------------------------
    >>> outputs = pipe.run(returns_matrix, asset_names=["SPY", "TLT", "GLD"])
    """

    def __init__(self, config: ModelingConfig):
        self.config = config
        self.outputs: dict = {}
        self._returns: Optional[np.ndarray] = None
        self._returns_matrix: Optional[np.ndarray] = None

    def run(
        self,
        returns: Union[np.ndarray, pd.Series, pd.DataFrame],
        asset_names: Optional[List[str]] = None,
    ) -> dict:
        """
        Chạy toàn bộ pipeline.

        Parameters
        ----------
        returns     : 1-D array/Series HOẶC 2-D matrix (n × d) cho copula
        asset_names : tên tài sản (nếu returns là 2-D)

        Returns
        -------
        dict với keys: dist_comparison, best_distribution, evt_summary,
                       var_comparison, cvar_comparison,
                       copula_comparison (nếu có)
        """
        # Chuẩn hóa input
        if isinstance(returns, pd.DataFrame):
            self._returns_matrix = returns.values
            self._returns = returns.iloc[:, 0].values  # dùng cột đầu cho phân phối
            asset_names = asset_names or list(returns.columns)
        elif isinstance(returns, pd.Series):
            self._returns = returns.dropna().values
            self._returns_matrix = None
        else:
            arr = np.asarray(returns)
            if arr.ndim == 2:
                self._returns_matrix = arr
                self._returns = arr[:, 0]
            else:
                self._returns = arr[~np.isnan(arr)]
                self._returns_matrix = None

        logger.info(f"ModelingPipeline: n_obs={len(self._returns)}")

        # Steps
        self._fit_distributions()
        if self.config.fit_evt:
            self._fit_evt()
        self._compare_var_cvar()
        if self.config.fit_copula and self._returns_matrix is not None:
            self._fit_copula(asset_names)

        if self.config.output_dir:
            self._save_outputs()

        logger.info("ModelingPipeline completed.")
        return self.outputs

    # ------------------------------------------------------------------
    # Step 1: Distribution fitting
    # ------------------------------------------------------------------

    def _fit_distributions(self) -> None:
        r = self._returns
        logger.info("Fitting distributions...")

        try:
            dist_df = fit_all(
                r,
                distributions=self.config.distributions,
                rank_by=self.config.rank_by,
            )
            self.outputs["dist_comparison"] = dist_df
            logger.info(f"  Best by {self.config.rank_by}: {dist_df.index[0]}")
        except Exception as e:
            logger.warning(f"  Distribution fit failed: {e}")
            dist_df = pd.DataFrame()
            self.outputs["dist_comparison"] = dist_df

        try:
            best = best_fit(r, rank_by=self.config.rank_by)
            self.outputs["best_distribution"] = best
            logger.info(f"  Best fit: {best.dist_name} (AIC={best.aic:.2f})")
        except Exception as e:
            logger.warning(f"  Best fit selection failed: {e}")
            self.outputs["best_distribution"] = None

    # ------------------------------------------------------------------
    # Step 2: EVT
    # ------------------------------------------------------------------

    def _fit_evt(self) -> None:
        r = self._returns
        logger.info("Fitting EVT (POT/GPD)...")
        try:
            gpd = fit_gpd_pot(r, threshold_quantile=self.config.threshold_quantile)
            self.outputs["gpd_fit"] = gpd
            logger.info(
                f"  GPD: ξ={gpd.xi:.4f}, σ={gpd.sigma:.4f}, "
                f"n_exc={gpd.n_exceedances}"
            )
        except Exception as e:
            logger.warning(f"  GPD fit failed: {e}")
            self.outputs["gpd_fit"] = None

        try:
            evt_sum = evt_summary(
                r,
                confidence_levels=self.config.confidence_levels,
                threshold_quantile=self.config.threshold_quantile,
            )
            self.outputs["evt_summary"] = evt_sum
        except Exception as e:
            logger.warning(f"  EVT summary failed: {e}")
            self.outputs["evt_summary"] = None

    # ------------------------------------------------------------------
    # Step 3: VaR / CVaR comparison
    # ------------------------------------------------------------------

    def _compare_var_cvar(self) -> None:
        r = self._returns
        logger.info("Comparing VaR / CVaR models...")

        for cl in self.config.confidence_levels:
            pct = int(cl * 100)
            try:
                var_df = compare_var_methods(r, confidence=cl)
                self.outputs[f"var_comparison_{pct}"] = var_df
            except Exception as e:
                logger.warning(f"  VaR comparison {pct}% failed: {e}")

            try:
                cvar_df = compare_cvar_methods(r, confidence=cl)
                self.outputs[f"cvar_comparison_{pct}"] = cvar_df
            except Exception as e:
                logger.warning(f"  CVaR comparison {pct}% failed: {e}")

        # Convenience: 95% as default key
        if "var_comparison_95" in self.outputs:
            self.outputs["var_comparison"] = self.outputs["var_comparison_95"]
            self.outputs["cvar_comparison"] = self.outputs.get("cvar_comparison_95")

    # ------------------------------------------------------------------
    # Step 4: Copula (multi-asset)
    # ------------------------------------------------------------------

    def _fit_copula(self, asset_names: Optional[List[str]] = None) -> None:
        R = self._returns_matrix
        if R is None or R.shape[1] < 2:
            logger.warning("Copula fit requires at least 2 assets.")
            return

        logger.info("Fitting copulas...")
        u = pseudo_observations(R)

        try:
            copula_df = fit_all_copulas(u, rank_by=self.config.rank_by)
            self.outputs["copula_comparison"] = copula_df
            logger.info(f"  Best copula: {copula_df.index[0]}")
        except Exception as e:
            logger.warning(f"  Copula fit failed: {e}")

        # Tail dependence matrix (bivariate pairs)
        if R.shape[1] == 2:
            try:
                lam_l, lam_u = empirical_tail_dependence(u, quantile=0.05)
                self.outputs["tail_dep_lower"] = lam_l
                self.outputs["tail_dep_upper"] = lam_u
                logger.info(f"  Tail dep: λ_L={lam_l:.3f}, λ_U={lam_u:.3f}")
            except Exception as e:
                logger.warning(f"  Tail dependence failed: {e}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_outputs(self) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        save_map = {
            "dist_comparison": "distribution_comparison.csv",
            "var_comparison": "var_comparison.csv",
            "cvar_comparison": "cvar_comparison.csv",
            "evt_summary": "evt_summary.csv",
            "copula_comparison": "copula_comparison.csv",
        }

        for key, fname in save_map.items():
            obj = self.outputs.get(key)
            if obj is None:
                continue
            path = out_dir / fname
            try:
                if isinstance(obj, pd.DataFrame):
                    obj.to_csv(path)
                elif isinstance(obj, pd.Series):
                    obj.to_frame("value").to_csv(path)
                logger.info(f"Saved: {path}")
            except Exception as e:
                logger.warning(f"  Could not save {fname}: {e}")

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def best_var(self, confidence: float = 0.95) -> float:
        """VaR từ best-fit distribution."""
        best: Optional[DistributionFit] = self.outputs.get("best_distribution")
        if best is None:
            raise RuntimeError("Run pipeline first.")
        from ..models.distribution import fitted_quantile
        return fitted_quantile(best, 1 - confidence)

    def evt_var(self, confidence: float = 0.99) -> float:
        """EVT-VaR từ GPD fit."""
        gpd: Optional[GPDResult] = self.outputs.get("gpd_fit")
        if gpd is None:
            raise RuntimeError("EVT not run or failed.")
        return gpd.var(confidence)

    def evt_cvar(self, confidence: float = 0.99) -> float:
        """EVT-CVaR từ GPD fit."""
        gpd: Optional[GPDResult] = self.outputs.get("gpd_fit")
        if gpd is None:
            raise RuntimeError("EVT not run or failed.")
        return gpd.cvar(confidence)