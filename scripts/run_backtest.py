#!/usr/bin/env python3
"""
scripts/run_backtest.py
------------------------
CLI entry point cho VaR/CVaR Backtesting.

Chức năng
---------
  - Rolling VaR forecast cho nhiều models
  - Kupiec POF test, Christoffersen Independence test
  - Conditional Coverage test
  - Basel Traffic Light system
  - McNeil-Frey ES backtest
  - So sánh tất cả models → bảng xếp hạng

Usage
-----
  # Chạy với config mặc định (SPY, confidence=99%)
  python scripts/run_backtest.py

  # Chỉ định ticker và confidence
  python scripts/run_backtest.py \\
      --ticker SPY \\
      --confidence 0.99 \\
      --window 252 \\
      --start 2010-01-01

  # Chỉ chạy một số model cụ thể
  python scripts/run_backtest.py \\
      --models hs parametric_t garch_normal filtered_hs \\
      --confidence 0.99

  # Từ CSV
  python scripts/run_backtest.py \\
      --csv data/raw/spy_returns.csv \\
      --confidence 0.95 \\
      --output-dir outputs/backtest_spy
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

from src.utils import load_config
from src.risk import (
    backtest_multiple_models,
    backtest_var_model,
    kupiec_test,
    christoffersen_test,
    conditional_coverage_test,
    basel_traffic_light,
    mcneil_frey_test,
    compute_violations,
    BacktestReport,
)
from src.models import (
    hs_var,
    parametric_var,
    cornish_fisher_var,
    garch_var,
    filtered_hs_var,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_backtest")

ALL_MODELS = ["hs", "parametric_normal", "parametric_t",
              "cornish_fisher", "garch_normal", "garch_t", "filtered_hs"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fat-Tail Risk – VaR Backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", "-c", default="configs/risk.yaml")
    parser.add_argument("--ticker", "-t", default=None,
                        help="Ticker yfinance (vd SPY, VNM)")
    parser.add_argument("--csv", default=None,
                        help="CSV file (index=date, cột đầu=return)")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument(
        "--confidence", type=float, default=None,
        help="Mức tin cậy VaR (default: 0.99 per Basel)",
    )
    parser.add_argument(
        "--window", type=int, default=None,
        help="Rolling window (ngày, default: 252)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=ALL_MODELS + ["all"],
        help="Models cần backtest (default: all)",
    )
    parser.add_argument(
        "--run-es-backtest", action="store_true",
        help="Chạy ES (CVaR) backtest thêm vào VaR",
    )
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_returns(args: argparse.Namespace, cfg: dict) -> pd.Series:
    if args.csv:
        logger.info(f"Loading from CSV: {args.csv}")
        df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        return df.iloc[:, 0].dropna().rename("return")

    ticker = args.ticker or cfg.get("data", {}).get("benchmark", "SPY")
    try:
        import yfinance as yf
        logger.info(f"Fetching {ticker} ({args.start} → {args.end})...")
        df = yf.download(ticker, start=args.start, end=args.end,
                         progress=False)
        prices = df["Adj Close"].dropna()
        returns = np.log(prices / prices.shift(1)).dropna()
        returns.name = "return"
        logger.info(f"Loaded {len(returns):,} observations")
        return returns
    except Exception as e:
        logger.warning(f"yfinance failed ({e}). Using synthetic data.")

    from src.simulation import gbm_returns
    r = gbm_returns(n_steps=252 * 15, n_paths=1, mu=0.07, sigma=0.20, seed=42)
    return pd.Series(r[:, 0], name="return")


# ---------------------------------------------------------------------------
# Rolling VaR forecast builder
# ---------------------------------------------------------------------------

MODEL_FN_MAP = {
    "hs":               lambda r, cl: hs_var(r, cl).var,
    "parametric_normal":lambda r, cl: parametric_var(r, cl, "normal").var,
    "parametric_t":     lambda r, cl: parametric_var(r, cl, "student_t").var,
    "cornish_fisher":   lambda r, cl: cornish_fisher_var(r, cl).var,
    "garch_normal":     lambda r, cl: garch_var(r, cl, dist="normal").var,
    "garch_t":          lambda r, cl: garch_var(r, cl, dist="student_t").var,
    "filtered_hs":      lambda r, cl: filtered_hs_var(r, cl).var,
}


def build_rolling_forecasts(
    r: np.ndarray,
    models: list,
    confidence: float,
    window: int,
) -> dict:
    """Tạo rolling VaR forecasts cho từng model."""
    n = len(r)
    forecasts = {}
    for name in models:
        fn = MODEL_FN_MAP.get(name)
        if fn is None:
            logger.warning(f"Unknown model: {name}")
            continue
        arr = np.full(n, np.nan)
        logger.info(f"  Building rolling forecasts: {name}...")
        for t in range(window, n):
            try:
                arr[t] = fn(r[t - window: t], confidence)
            except Exception:
                pass
        forecasts[name] = arr
    return forecasts


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

TRAFFIC_COLORS = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "reset":  "\033[0m",
}


def traffic_light_str(tl: str) -> str:
    color = TRAFFIC_COLORS.get(tl, "")
    reset = TRAFFIC_COLORS["reset"]
    return f"{color}●{reset} {tl.upper()}"


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def print_backtest_report(report: BacktestReport, confidence: float) -> None:
    df = report.summary()
    print(f"\n  Confidence: {int(confidence*100)}%  |  N obs: {report.n_obs:,}")
    print(f"  Expected violation rate: {(1-confidence):.2%}\n")

    header = (f"  {'Model':<22} {'N_viol':>7} {'Viol%':>7} "
              f"{'KupiecP':>9} {'ChrP':>9} {'Basel':>10} {'Pass?':>7}")
    print(header)
    print("  " + "-" * 72)

    for model, row in df.iterrows():
        pass_str = "✓" if row.get("pass_kupiec") and row.get("pass_chr") else "✗"
        tl = row.get("traffic_light", "")
        tl_str = traffic_light_str(tl) if tl else ""
        print(
            f"  {model:<22} "
            f"{int(row.get('n_violations', 0)):>7} "
            f"{row.get('violation_rate', 0):>7.2%} "
            f"{row.get('kupiec_pvalue', np.nan):>9.4f} "
            f"{row.get('chr_pvalue', np.nan):>9.4f} "
            f"  {tl_str:<14} "
            f"{pass_str:>3}"
        )

    print(f"\n  Best model: {report.best_model()}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_backtest_figures(
    returns: pd.Series,
    forecasts: dict,
    best_model: str,
    confidence: float,
    window: int,
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.visualization import plot_backtest_violations

        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        valid_start = window
        r_valid = returns.iloc[valid_start:]

        for model_name, var_arr in forecasts.items():
            var_valid = pd.Series(
                var_arr[valid_start:],
                index=r_valid.index,
                name="var_forecast",
            )
            try:
                fig = plot_backtest_violations(
                    r_valid, var_valid,
                    model_name=model_name,
                    confidence=confidence,
                )
                fname = fig_dir / f"backtest_{model_name}.png"
                fig.savefig(fname, bbox_inches="tight", dpi=120)
                plt.close(fig)
                logger.info(f"  Saved: {fname}")
            except Exception as e:
                logger.warning(f"  Figure '{model_name}' failed: {e}")

    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("  Fat-Tail Risk – run_backtest.py")
    logger.info("=" * 60)

    # ── 1. Config ────────────────────────────────────────────────────
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    bt_cfg = cfg.get("backtest", {})

    confidence = (
        args.confidence
        or bt_cfg.get("confidence", 0.99)
    )
    window = (
        args.window
        or bt_cfg.get("rolling_window", 252)
    )
    models = (
        args.models if args.models and "all" not in (args.models or [])
        else bt_cfg.get("models", ALL_MODELS)
    )
    output_dir = Path(
        args.output_dir
        or cfg.get("output", {}).get("dir", "outputs/backtest")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Confidence  : {confidence:.0%}")
    logger.info(f"Window      : {window} days")
    logger.info(f"Models      : {models}")
    logger.info(f"Output dir  : {output_dir}")

    # ── 2. Data ──────────────────────────────────────────────────────
    returns_series = load_returns(args, cfg)
    r = returns_series.values
    n = len(r)

    if n < window + 50:
        logger.error(f"Too few observations ({n}) for window={window}. Aborting.")
        sys.exit(1)

    logger.info(f"Returns     : {n:,} observations")

    # ── 3. Build rolling forecasts ───────────────────────────────────
    print_section("Building Rolling VaR Forecasts")
    forecasts = build_rolling_forecasts(r, models, confidence, window)

    # ── 4. Backtest ──────────────────────────────────────────────────
    valid_start = window
    r_valid = r[valid_start:]
    forecasts_valid = {k: v[valid_start:] for k, v in forecasts.items()}

    print_section("VaR Backtest Results")
    try:
        report = backtest_multiple_models(
            r_valid, forecasts_valid, confidence=confidence
        )
        print_backtest_report(report, confidence)

        # Save CSV
        path = output_dir / "backtest_summary.csv"
        report.summary().to_csv(path)
        logger.info(f"\nSaved: {path}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)

    # ── 5. Individual test detail (best model) ────────────────────────
    best = report.best_model()
    print_section(f"Detail: {best}")
    if best in forecasts_valid:
        violations = compute_violations(r_valid, forecasts_valid[best])
        n_viol = int(violations.sum())

        lr_uc, p_uc = kupiec_test(violations, confidence)
        lr_ind, p_ind = christoffersen_test(violations)
        lr_cc, p_cc = conditional_coverage_test(violations, confidence)
        tl = basel_traffic_light(n_viol, n_obs=len(r_valid), confidence=confidence)

        print(f"  Violations        : {n_viol:,} / {len(r_valid):,} "
              f"({n_viol/len(r_valid):.2%})")
        print(f"  Expected          : {(1-confidence):.2%}")
        print(f"  Kupiec POF        : stat={lr_uc:.3f}, p={p_uc:.4f} "
              f"{'✓ PASS' if p_uc > 0.05 else '✗ FAIL'}")
        print(f"  Christoffersen    : stat={lr_ind:.3f}, p={p_ind:.4f} "
              f"{'✓ PASS' if p_ind > 0.05 else '✗ FAIL'}")
        print(f"  Cond. Coverage    : stat={lr_cc:.3f}, p={p_cc:.4f} "
              f"{'✓ PASS' if p_cc > 0.05 else '✗ FAIL'}")
        print(f"  Basel Traffic     : {traffic_light_str(tl)}")

    # ── 6. ES backtest (tùy chọn) ─────────────────────────────────────
    if args.run_es_backtest and "hs" in forecasts_valid:
        print_section("ES (CVaR) Backtest – McNeil-Frey Test")
        try:
            from src.models import historical_cvar as hist_cvar
            # Build rolling CVaR forecasts
            cvar_arr = np.full(len(r_valid), np.nan)
            for t in range(0, len(r_valid)):
                train_end = valid_start + t
                if train_end < window:
                    continue
                train = r[train_end - window: train_end]
                try:
                    cvar_arr[t] = hist_cvar(train, confidence).cvar
                except Exception:
                    pass

            t_stat, p_val = mcneil_frey_test(
                r_valid,
                forecasts_valid["hs"],
                cvar_arr,
                confidence=confidence,
            )
            print(f"  McNeil-Frey t-stat: {t_stat:.4f}")
            print(f"  p-value           : {p_val:.4f} "
                  f"{'✓ ES calibrated' if p_val > 0.05 else '✗ ES mis-specified'}")
        except Exception as e:
            logger.warning(f"ES backtest failed: {e}")

    # ── 7. Figures ───────────────────────────────────────────────────
    if not args.no_figures:
        save_backtest_figures(
            returns_series, forecasts,
            best, confidence, window, output_dir,
        )

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.1f}s → {output_dir}")


if __name__ == "__main__":
    main()