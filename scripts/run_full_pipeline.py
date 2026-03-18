#!/usr/bin/env python3
"""
scripts/run_full_pipeline.py
-----------------------------
CLI entry point chạy toàn bộ Fat-Tail-Risk pipeline end-to-end.

Pipeline stages
---------------
  1. SimulationPipeline  – Monte Carlo scenarios
  2. ModelingPipeline    – distribution fit, EVT, copula
  3. RiskPipeline        – metrics, stress test, backtest
  4. Visualization       – tất cả figures
  5. Report              – HTML + PDF report

Usage
-----
  # Chạy đầy đủ với SPY, 1 lệnh duy nhất
  python scripts/run_full_pipeline.py

  # Chỉ định ticker và output
  python scripts/run_full_pipeline.py \\
      --ticker SPY \\
      --start 2010-01-01 \\
      --output-dir results/spy_2024 \\
      --n-paths 50000

  # Chạy nhanh (ít paths hơn, bỏ report PDF)
  python scripts/run_full_pipeline.py \\
      --ticker SPY \\
      --n-paths 5000 \\
      --no-pdf \\
      --quiet

  # Chạy nhiều tickers
  python scripts/run_full_pipeline.py \\
      --ticker SPY --n-paths 20000 --output-dir results/spy
  python scripts/run_full_pipeline.py \\
      --ticker VNM --n-paths 20000 --output-dir results/vnm

  # Từ CSV
  python scripts/run_full_pipeline.py \\
      --csv data/raw/portfolio_returns.csv \\
      --name portfolio_q4 \\
      --output-dir results/portfolio_q4
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

from src.utils import (
    load_config,
    namespace,
    build_simulation_config,
    build_modeling_config,
    build_risk_config,
)
from src.pipelines import (
    FullPipeline,
    FullPipelineConfig,
    SimulationConfig,
    ModelingConfig,
    RiskConfig,
    run_full_analysis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_full_pipeline")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fat-Tail Risk – Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data input
    parser.add_argument("--ticker", "-t", default=None,
                        help="Ticker yfinance (vd SPY, VNM, BTC-USD)")
    parser.add_argument("--csv", default=None,
                        help="CSV file return (index=date)")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--name", default=None,
                        help="Tên phân tích (dùng trong report/output)")

    # Configs
    parser.add_argument("--sim-config",   default="configs/simulation.yaml")
    parser.add_argument("--model-config", default="configs/evt.yaml")
    parser.add_argument("--risk-config",  default="configs/risk.yaml")

    # Override key params
    parser.add_argument("--n-paths", type=int, default=None,
                        help="Số Monte Carlo paths")
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Kịch bản MC (vd: base gfc_2008 covid_2020 black_swan)",
    )
    parser.add_argument("--risk-free", type=float, default=None,
                        help="Lãi suất phi rủi ro annualised (vd: 0.05)")
    parser.add_argument("--portfolio-value", type=float, default=None,
                        help="Giá trị danh mục (USD)")
    parser.add_argument("--seed", type=int, default=42)

    # Stages on/off
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-modeling",   action="store_true")
    parser.add_argument("--skip-risk",       action="store_true")
    parser.add_argument("--no-report",       action="store_true",
                        help="Không tạo HTML/PDF report")
    parser.add_argument("--no-pdf",          action="store_true",
                        help="Tạo HTML nhưng không tạo PDF")

    # Output
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_returns(args: argparse.Namespace) -> tuple[pd.Series, str]:
    """Load returns, trả về (series, name)."""
    name = args.name

    # CSV
    if args.csv:
        path = Path(args.csv)
        logger.info(f"Loading from CSV: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        r = df.iloc[:, 0].dropna().rename("return")
        name = name or path.stem
        logger.info(f"Loaded {len(r):,} obs from {path.name}")
        return r, name

    # yfinance
    ticker = args.ticker or "SPY"
    name = name or ticker
    try:
        import yfinance as yf
        logger.info(f"Fetching {ticker} ({args.start} → {args.end})...")
        df = yf.download(ticker, start=args.start, end=args.end,
                         progress=False)
        if df.empty:
            raise ValueError("Empty dataframe from yfinance")
        prices = df["Adj Close"].dropna()
        r = np.log(prices / prices.shift(1)).dropna()
        r.name = "return"
        logger.info(f"Loaded {len(r):,} observations for {ticker}")
        return r, name
    except Exception as e:
        logger.warning(f"yfinance failed ({e}). Using synthetic GBM data.")
        name = name or "synthetic_gbm"

    # Synthetic
    from src.simulation import gbm_returns
    r_arr = gbm_returns(n_steps=252 * 15, n_paths=1, mu=0.07,
                        sigma=0.20, seed=args.seed)
    dates = pd.date_range("2005-01-01", periods=252 * 15, freq="B")
    r = pd.Series(r_arr[:, 0], index=dates[:len(r_arr)], name="return")
    logger.info(f"Generated {len(r):,} synthetic GBM returns")
    return r, name


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_full_config(args: argparse.Namespace, name: str) -> FullPipelineConfig:
    """Xây dựng FullPipelineConfig từ CLI args + YAML configs."""

    # Load sub-configs
    sim_cfg_dict  = load_config(args.sim_config)   if Path(args.sim_config).exists()   else {}
    model_cfg_dict = load_config(args.model_config) if Path(args.model_config).exists() else {}
    risk_cfg_dict  = load_config(args.risk_config)  if Path(args.risk_config).exists()  else {}

    # Override từ CLI
    if args.n_paths:
        sim_cfg_dict.setdefault("monte_carlo", {})["n_paths"] = args.n_paths
    if args.scenarios:
        sim_cfg_dict.setdefault("scenarios", {})["active"] = args.scenarios
    if args.seed:
        sim_cfg_dict.setdefault("monte_carlo", {})["seed"] = args.seed
    if args.risk_free is not None:
        risk_cfg_dict.setdefault("metrics", {})["risk_free_annual"] = args.risk_free
    if args.portfolio_value:
        risk_cfg_dict.setdefault("portfolio", {})["value"] = args.portfolio_value

    # Build dataclasses
    sim_config   = build_simulation_config(sim_cfg_dict)
    model_config = build_modeling_config(model_cfg_dict)
    risk_config  = build_risk_config(risk_cfg_dict)

    output_dir = args.output_dir or f"outputs/{name}"

    return FullPipelineConfig(
        name=name,
        output_dir=output_dir,
        run_simulation=not args.skip_simulation,
        run_modeling=not args.skip_modeling,
        run_risk=not args.skip_risk,
        run_report=not args.no_report,
        sim_config=sim_config,
        model_config=model_config,
        risk_config=risk_config,
    )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_final_summary(pipe: FullPipeline, elapsed: float) -> None:
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)

    # Risk metrics
    rm = pipe.risk_metrics
    if rm:
        print(f"\n  ── Risk Metrics ────────────────────────────────")
        print(f"  Ann. Return      : {rm.annualised_return:.2%}")
        print(f"  Ann. Volatility  : {rm.annualised_vol:.2%}")
        print(f"  Sharpe Ratio     : {rm.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio    : {rm.sortino_ratio:.2f}")
        print(f"  Calmar Ratio     : {rm.calmar_ratio:.2f}")
        print(f"  Max Drawdown     : {rm.max_drawdown:.2%}")
        print(f"  VaR 95%          : {rm.var_95:.4f}")
        print(f"  CVaR 95%         : {rm.cvar_95:.4f}")
        print(f"  VaR 99%          : {rm.var_99:.4f}")
        print(f"  CVaR 99%         : {rm.cvar_99:.4f}")
        if not np.isnan(rm.evt_var_99):
            print(f"  EVT-VaR 99%      : {rm.evt_var_99:.4f}")
        print(f"  Skewness         : {rm.skewness:.3f}")
        print(f"  Excess Kurtosis  : {rm.excess_kurtosis:.3f}")

    # Best distribution
    best_dist = pipe.best_distribution
    if best_dist:
        print(f"\n  ── Best Distribution ───────────────────────────")
        print(f"  {best_dist.dist_name.upper()}")
        for k, v in best_dist.params.items():
            print(f"    {k:<12}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        print(f"  AIC: {best_dist.aic:.2f}  |  BIC: {best_dist.bic:.2f}")

    # Best backtest model
    bt = pipe.backtest_report
    if bt:
        print(f"\n  ── Backtest ────────────────────────────────────")
        print(f"  Best VaR model   : {bt.best_model()}")
        summary = bt.summary()
        for model, row in summary.iterrows():
            tl = row.get("traffic_light", "")
            tl_indicator = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(tl, "⚪")
            pass_str = "✓" if row.get("pass_kupiec") else "✗"
            print(f"  {pass_str} {tl_indicator} {model:<22} "
                  f"violations={int(row.get('n_violations', 0))}  "
                  f"p={row.get('kupiec_pvalue', np.nan):.4f}")

    # Scenario comparison
    sc = pipe.scenario_comparison
    if sc is not None:
        print(f"\n  ── Scenario Comparison ─────────────────────────")
        cols = [c for c in ["mean_return", "var_95", "cvar_95", "min_return"]
                if c in sc.columns]
        print(sc[cols].round(4).to_string())

    print(f"\n  {'─' * 50}")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"  Output dir   : {pipe.config.output_dir}")

    # Report paths
    report = pipe.outputs.get("report")
    if report:
        out = Path(pipe.config.output_dir)
        html = out / f"{pipe.config.name}_report.html"
        pdf  = out / f"{pipe.config.name}_report.pdf"
        if html.exists():
            print(f"  HTML report  : {html}")
        if pdf.exists():
            print(f"  PDF  report  : {pdf}")

    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    t0 = time.time()
    logger.info("=" * 65)
    logger.info("  Fat-Tail Risk – run_full_pipeline.py")
    logger.info("=" * 65)

    # ── 1. Load data ─────────────────────────────────────────────────
    returns, name = load_returns(args)
    logger.info(f"Name        : {name}")
    logger.info(f"N obs       : {len(returns):,}")
    logger.info(f"Date range  : {returns.index[0].date()} → {returns.index[-1].date()}")

    # ── 2. Build config ──────────────────────────────────────────────
    full_cfg = build_full_config(args, name)
    logger.info(f"Output dir  : {full_cfg.output_dir}")
    logger.info(f"Stages      : simulation={full_cfg.run_simulation}, "
                f"modeling={full_cfg.run_modeling}, "
                f"risk={full_cfg.run_risk}, "
                f"report={full_cfg.run_report}")

    # ── 3. Run pipeline ──────────────────────────────────────────────
    pipe = FullPipeline(full_cfg)
    pipe.run(returns)

    # ── 4. Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    if not args.quiet:
        print_final_summary(pipe, elapsed)
    else:
        logger.info(f"Done in {elapsed:.1f}s → {full_cfg.output_dir}")


if __name__ == "__main__":
    main()