#!/usr/bin/env python3
"""
scripts/run_evt.py
-------------------
CLI entry point cho Extreme Value Theory (EVT) analysis.

Chức năng
---------
  - Fit GPD (POT) và GEV (Block Maxima) lên chuỗi return
  - Tính EVT-VaR, EVT-CVaR, Return Levels
  - Vẽ: Hill plot, Mean Excess Function, Threshold Stability,
         Q-Q plot, Return Level plot, Tail fit

Usage
-----
  # Chạy với config mặc định
  python scripts/run_evt.py

  # Chỉ định ticker và ngưỡng
  python scripts/run_evt.py \\
      --ticker SPY \\
      --start 2005-01-01 \\
      --threshold-quantile 0.90 \\
      --confidence 0.95 0.99 0.999

  # Từ file CSV
  python scripts/run_evt.py \\
      --csv data/raw/spy_returns.csv \\
      --threshold-quantile 0.92 \\
      --output-dir outputs/evt_spy

  # Chạy cả Block Maxima
  python scripts/run_evt.py --method both --block-size 21
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, build_modeling_config
from src.models import (
    fit_gpd_pot,
    fit_gev,
    evt_var,
    evt_cvar,
    return_levels,
    evt_summary,
    threshold_stability_plot_data,
    mean_excess_plot_data,
)
from src.features import hill_estimator, tail_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_evt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fat-Tail Risk – EVT Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", "-c", default="configs/evt.yaml")
    parser.add_argument("--ticker", "-t", default=None,
                        help="Ticker yfinance (vd SPY, VNM)")
    parser.add_argument("--csv", default=None,
                        help="Đường dẫn CSV file (index=date, cột đầu=return)")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument(
        "--method", choices=["pot", "bm", "both"], default="both",
        help="Phương pháp EVT: pot | bm | both (default: both)",
    )
    parser.add_argument(
        "--threshold-quantile", "-q", type=float, default=None,
        help="Phân vị ngưỡng POT (override config, default 0.90)",
    )
    parser.add_argument(
        "--block-size", type=int, default=None,
        help="Kích thước block cho GEV (default 21 = monthly)",
    )
    parser.add_argument(
        "--confidence", nargs="+", type=float, default=None,
        help="Mức tin cậy cho VaR/CVaR (vd: 0.95 0.99 0.999)",
    )
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quiet", "-q2", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_returns(args: argparse.Namespace, cfg: dict) -> pd.Series:
    """Load returns từ CSV, yfinance, hoặc synthetic."""
    # CSV
    if args.csv:
        path = Path(args.csv)
        logger.info(f"Loading from CSV: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.iloc[:, 0].dropna().rename("return")

    # yfinance
    ticker = args.ticker or cfg.get("data", {}).get("benchmark", "SPY")
    try:
        import yfinance as yf
        logger.info(f"Fetching {ticker} ({args.start} → {args.end})...")
        df = yf.download(ticker, start=args.start, end=args.end, progress=False)
        prices = df["Adj Close"].dropna()
        returns = np.log(prices / prices.shift(1)).dropna()
        returns.name = "return"
        logger.info(f"Loaded {len(returns):,} observations")
        return returns
    except Exception as e:
        logger.warning(f"yfinance failed ({e}). Using synthetic data.")

    # Synthetic
    from src.simulation import gbm_returns
    r = gbm_returns(n_steps=252 * 10, n_paths=1, mu=0.07, sigma=0.20, seed=42)
    return pd.Series(r[:, 0], name="return")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_evt_figures(
    returns: np.ndarray,
    gpd_result,
    output_dir: Path,
    threshold_q: float,
) -> None:
    """Lưu toàn bộ EVT diagnostic figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.visualization import (
            plot_hill,
            plot_mean_excess,
            plot_threshold_stability,
            plot_evt_tail,
            plot_var_cvar,
        )

        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        figures = {
            "hill_plot": lambda: plot_hill(returns),
            "mean_excess": lambda: plot_mean_excess(returns),
            "threshold_stability": lambda: plot_threshold_stability(returns),
            "evt_tail_fit": lambda: plot_evt_tail(returns, threshold_quantile=threshold_q),
            "var_cvar_dist": lambda: plot_var_cvar(returns, confidence=0.99),
        }

        for name, fn in figures.items():
            try:
                fig = fn()
                fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight", dpi=120)
                plt.close(fig)
                logger.info(f"  Saved: {fig_dir / name}.png")
            except Exception as e:
                logger.warning(f"  Figure '{name}' failed: {e}")

    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    width = 55
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_gpd(gpd) -> None:
    print(f"  Shape  ξ      : {gpd.xi:.4f}  {'(fat-tail)' if gpd.xi > 0 else '(thin-tail)'}")
    print(f"  Scale  σ      : {gpd.sigma:.4f}")
    print(f"  Threshold  u  : {gpd.threshold:.4f}")
    print(f"  N exceedances : {gpd.n_exceedances:,}  "
          f"({gpd.exceedance_rate:.2%} of {gpd.n_total:,})")
    print(f"  AIC           : {gpd.aic:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    t0 = time.time()
    logger.info("=" * 55)
    logger.info("  Fat-Tail Risk – run_evt.py")
    logger.info("=" * 55)

    # ── 1. Config ────────────────────────────────────────────────────
    cfg = load_config(args.config) if Path(args.config).exists() else {}

    threshold_q = (
        args.threshold_quantile
        or cfg.get("pot", {}).get("threshold", {}).get("percentile", 0.90)
    )
    block_size = (
        args.block_size
        or cfg.get("block_maxima", {}).get("block_size", 21)
    )
    confidence_levels = (
        args.confidence
        or cfg.get("evt_risk", {}).get("confidence_levels", [0.95, 0.99, 0.999])
    )
    output_dir = Path(
        args.output_dir
        or cfg.get("output", {}).get("dir", "outputs/evt")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Threshold quantile : {threshold_q:.0%}")
    logger.info(f"Confidence levels  : {confidence_levels}")
    logger.info(f"Output dir         : {output_dir}")

    # ── 2. Data ──────────────────────────────────────────────────────
    returns_series = load_returns(args, cfg)
    r = returns_series.values

    # ── 3. Basic tail features ───────────────────────────────────────
    print_section("Tail Features Summary")
    ts = tail_summary(r, confidence_levels=tuple(confidence_levels))
    for key, val in ts.items():
        print(f"  {key:<30}: {val:.4f}" if isinstance(val, float)
              else f"  {key:<30}: {val}")

    # ── 4. POT / GPD ─────────────────────────────────────────────────
    if args.method in ("pot", "both"):
        print_section("POT / GPD Fit")
        try:
            gpd = fit_gpd_pot(r, threshold_quantile=threshold_q)
            print_gpd(gpd)

            print("\n  EVT Risk Measures:")
            for cl in confidence_levels:
                v = gpd.var(cl)
                cv = gpd.cvar(cl)
                print(f"    VaR  {int(cl*100):3d}%: {v:.4f}  |  "
                      f"CVaR {int(cl*100):3d}%: {cv:.4f}")

            print("\n  Return Levels (years → max loss):")
            rl_df = return_levels(r, threshold_quantile=threshold_q)
            for period, row in rl_df.iterrows():
                print(f"    {period:4.0f}yr : {row['return_level_pct']:.2f}%")

        except Exception as e:
            logger.error(f"GPD fit failed: {e}")

    # ── 5. Block Maxima / GEV ────────────────────────────────────────
    if args.method in ("bm", "both"):
        print_section("Block Maxima / GEV Fit")
        try:
            gev = fit_gev(r, block_size=block_size)
            print(f"  Shape  ξ      : {gev.xi:.4f}")
            print(f"  Location  μ   : {gev.mu:.4f}")
            print(f"  Scale  σ      : {gev.sigma:.4f}")
            print(f"  N blocks      : {gev.n_blocks}")
            print(f"  AIC           : {gev.aic:.2f}")
            print(f"\n  GEV Return Levels:")
            for T in [1, 5, 10, 25, 50]:
                rl = gev.return_level(T)
                print(f"    {T:3d}yr: {rl:.4f}")
        except Exception as e:
            logger.error(f"GEV fit failed: {e}")

    # ── 6. Full EVT summary ──────────────────────────────────────────
    print_section("EVT Summary")
    try:
        evt_sum = evt_summary(
            r,
            confidence_levels=tuple(confidence_levels),
            threshold_quantile=threshold_q,
        )
        print(evt_sum.round(4).to_string())

        # Lưu CSV
        path = output_dir / "evt_summary.csv"
        evt_sum.to_frame("value").to_csv(path)
        logger.info(f"Saved: {path}")
    except Exception as e:
        logger.error(f"EVT summary failed: {e}")

    # ── 7. Hill estimator ────────────────────────────────────────────
    print_section("Hill Estimator (Tail Index)")
    try:
        alpha = hill_estimator(r, k_fraction=0.10)
        print(f"  Tail index α (Hill, k=10%): {alpha:.3f}")
        print(f"  Interpretation: ", end="")
        if alpha < 2:
            print("Infinite variance (extreme fat-tail)")
        elif alpha < 3:
            print("Infinite skewness (very fat-tail)")
        elif alpha < 4:
            print("Finite skewness, infinite kurtosis (fat-tail)")
        else:
            print("Finite kurtosis (moderate fat-tail)")
    except Exception as e:
        logger.warning(f"Hill estimator failed: {e}")

    # ── 8. Figures ───────────────────────────────────────────────────
    if not args.no_figures and args.method in ("pot", "both"):
        try:
            gpd = fit_gpd_pot(r, threshold_quantile=threshold_q)
            save_evt_figures(r, gpd, output_dir, threshold_q)
        except Exception:
            pass

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.1f}s → {output_dir}")


if __name__ == "__main__":
    main()