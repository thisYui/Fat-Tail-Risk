#!/usr/bin/env python3
"""
scripts/run_simulation.py
--------------------------
CLI entry point cho SimulationPipeline.

Chức năng
---------
  - Load config từ configs/simulation.yaml (hoặc file tùy chỉnh)
  - Fetch hoặc tạo dữ liệu return
  - Chạy Monte Carlo simulation cho các kịch bản
  - Lưu kết quả và figures

Usage
-----
  # Chạy với config mặc định
  python scripts/run_simulation.py

  # Chỉ định config và override tham số
  python scripts/run_simulation.py \\
      --config configs/simulation.yaml \\
      --n-paths 100000 \\
      --scenarios base gfc_2008 covid_2020 black_swan \\
      --output-dir outputs/sim_run1 \\
      --seed 42

  # Chỉ chạy một kịch bản cụ thể
  python scripts/run_simulation.py --scenarios gfc_2008 --n-paths 50000

  # Bật portfolio simulation
  python scripts/run_simulation.py --portfolio --n-paths 20000
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Đảm bảo import được src package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, namespace, build_simulation_config
from src.pipelines import SimulationPipeline, SimulationConfig
from src.simulation import (
    MonteCarloEngine,
    list_scenarios,
    CRISIS_SCENARIOS,
    summarise_mc,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_simulation")


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fat-Tail Risk – Monte Carlo Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/simulation.yaml",
        help="Đường dẫn file config YAML (default: configs/simulation.yaml)",
    )

    # Monte Carlo params (override config)
    parser.add_argument(
        "--n-paths", "-n",
        type=int,
        default=None,
        help="Số Monte Carlo paths (override config)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Số bước thời gian (override config, default 252 = 1 năm)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (override config)",
    )
    parser.add_argument(
        "--S0",
        type=float,
        default=None,
        help="Giá / giá trị ban đầu (default: 100.0)",
    )

    # Scenarios
    parser.add_argument(
        "--scenarios", "-s",
        nargs="+",
        default=None,
        help="Danh sách kịch bản (vd: base gfc_2008 covid_2020). "
             "Dùng --list-scenarios để xem tất cả.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Liệt kê tất cả kịch bản có sẵn rồi thoát",
    )

    # Portfolio
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Bật portfolio simulation (multi-asset)",
    )

    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Thư mục lưu kết quả (override config)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Không vẽ và lưu figures",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Giảm log output",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loader (demo: sinh synthetic hoặc đọc CSV)
# ---------------------------------------------------------------------------

def load_or_generate_returns(cfg: dict) -> pd.Series:
    """
    Load return từ file CSV hoặc sinh GBM synthetic.

    Ưu tiên: CSV file → yfinance → synthetic GBM.
    """
    data_cfg = cfg.get("data", {})
    raw_dir = Path(cfg.get("paths", {}).get("raw_dir", "data/raw"))

    # Thử đọc CSV
    csv_candidates = list(raw_dir.glob("*.csv"))
    if csv_candidates:
        path = csv_candidates[0]
        logger.info(f"Loading returns from {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        col = df.columns[0]
        return df[col].dropna().rename("return")

    # Thử yfinance
    try:
        import yfinance as yf
        ticker = data_cfg.get("benchmark", "SPY")
        start  = data_cfg.get("default_start", "2010-01-01")
        end    = data_cfg.get("default_end", "2024-12-31")
        logger.info(f"Fetching {ticker} from yfinance ({start} → {end})...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        prices = df["Adj Close"].dropna()
        returns = np.log(prices / prices.shift(1)).dropna()
        returns.name = "return"
        logger.info(f"Loaded {len(returns)} observations for {ticker}")
        return returns
    except Exception as e:
        logger.warning(f"yfinance failed ({e}). Using synthetic GBM data.")

    # Synthetic GBM
    from src.simulation import gbm_returns
    mc_cfg = cfg.get("monte_carlo", {})
    seed   = mc_cfg.get("seed", 42)
    n      = mc_cfg.get("n_steps", 252) * 5   # 5 năm synthetic
    r = gbm_returns(n_steps=n, n_paths=1, mu=0.07, sigma=0.20, seed=seed)
    returns = pd.Series(r[:, 0], name="return")
    logger.info(f"Generated {n} synthetic GBM returns (seed={seed})")
    return returns


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_figures(engine: MonteCarloEngine, output_dir: Path) -> None:
    """Vẽ và lưu figures cho các kịch bản đã chạy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.visualization import plot_scenario_fan, plot_scenario_comparison
        from src.simulation import compare_scenarios_summary

        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        cached = engine.cached_scenarios
        if not cached:
            return

        # Fan chart cho từng kịch bản
        for name in cached:
            result = engine._cache[name]
            fig = plot_scenario_fan(
                result.paths,
                S0=result.paths[0, 0],
                title=f"Monte Carlo Fan Chart – {name}",
            )
            fig.savefig(fig_dir / f"fan_{name}.png", bbox_inches="tight", dpi=120)
            plt.close(fig)

        # So sánh tất cả kịch bản
        results_dict = {name: engine._cache[name] for name in cached}
        comp = compare_scenarios_summary(results_dict)
        fig = plot_scenario_comparison(comp, metric="cvar_95",
                                       title="CVaR 95% – Scenario Comparison")
        fig.savefig(fig_dir / "scenario_comparison_cvar95.png",
                    bbox_inches="tight", dpi=120)
        plt.close(fig)

        logger.info(f"Figures saved to {fig_dir}")

    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # List scenarios and exit
    if args.list_scenarios:
        print("\nAvailable scenarios:")
        print(list_scenarios().to_string())
        print()
        sys.exit(0)

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    t0 = time.time()
    logger.info("=" * 55)
    logger.info("  Fat-Tail Risk – run_simulation.py")
    logger.info("=" * 55)

    # ── 1. Load config ───────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    overrides = {}
    if args.n_paths:
        overrides["monte_carlo.n_paths"] = args.n_paths
    if args.n_steps:
        overrides["monte_carlo.n_steps"] = args.n_steps
    if args.seed is not None:
        overrides["monte_carlo.seed"] = args.seed
    if args.S0:
        overrides["monte_carlo.S0"] = args.S0
    if args.output_dir:
        overrides["output.dir"] = args.output_dir
    if args.portfolio:
        overrides["portfolio.enabled"] = True

    cfg = load_config(config_path, overrides=overrides if overrides else None)
    logger.info(f"Config loaded: {config_path}")

    # ── 2. Build SimulationConfig ────────────────────────────────────
    sim_cfg = build_simulation_config(cfg)
    if args.scenarios:
        sim_cfg.scenarios = args.scenarios

    logger.info(f"Scenarios   : {sim_cfg.scenarios}")
    logger.info(f"n_paths     : {sim_cfg.n_paths:,}")
    logger.info(f"n_steps     : {sim_cfg.n_steps}")
    logger.info(f"seed        : {sim_cfg.seed}")
    logger.info(f"output_dir  : {sim_cfg.output_dir}")

    # ── 3. Load returns (cho bootstrap scenarios) ────────────────────
    returns = load_or_generate_returns(cfg)
    logger.info(f"Returns loaded: {len(returns):,} observations")

    # ── 4. Run SimulationPipeline ────────────────────────────────────
    pipe = SimulationPipeline(sim_cfg)
    outputs = pipe.run()

    # ── 5. Print summary ─────────────────────────────────────────────
    comp = outputs.get("scenario_comparison")
    if comp is not None:
        cols = [c for c in ["mean_return", "std_return", "var_95", "cvar_95", "min_return"]
                if c in comp.columns]
        print("\n" + "=" * 55)
        print("  Scenario Comparison")
        print("=" * 55)
        print(comp[cols].round(4).to_string())
        print()

    summaries = outputs.get("mc_summaries", {})
    for name, s in summaries.items():
        print(f"[{name}] mean={s.get('mean', np.nan):.4f} | "
              f"var_95={s.get('var_95', np.nan):.4f} | "
              f"cvar_95={s.get('cvar_95', np.nan):.4f}")

    # ── 6. Figures ───────────────────────────────────────────────────
    if not args.no_figures:
        save_figures(pipe.engine, Path(sim_cfg.output_dir))

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.1f}s → {sim_cfg.output_dir}")


if __name__ == "__main__":
    main()