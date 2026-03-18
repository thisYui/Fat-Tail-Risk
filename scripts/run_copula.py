#!/usr/bin/env python3
"""
scripts/run_copula.py
----------------------
CLI entry point cho Copula analysis – cấu trúc phụ thuộc đa tài sản.

Chức năng
---------
  - Fetch returns của nhiều tài sản
  - Chuyển sang pseudo-observations
  - Fit và so sánh Gaussian / Student-t / Clayton / Gumbel / Frank copula
  - Tính tail dependence matrix (λ_L, λ_U)
  - Sinh mẫu từ copula tốt nhất
  - Lưu figures: scatter, heatmap, density contour

Usage
-----
  # Chạy với cặp mặc định SPY-TLT
  python scripts/run_copula.py

  # Chỉ định tickers
  python scripts/run_copula.py \\
      --tickers SPY TLT GLD \\
      --start 2010-01-01

  # Chỉ phân tích một cặp cụ thể
  python scripts/run_copula.py \\
      --tickers SPY EEM \\
      --families gaussian student_t clayton \\
      --tail-quantile 0.05

  # Từ CSV
  python scripts/run_copula.py \\
      --csv data/raw/multi_returns.csv \\
      --output-dir outputs/copula_multi
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
from src.models import (
    pseudo_observations,
    fit_gaussian_copula,
    fit_student_t_copula,
    fit_archimedean_copula,
    fit_all_copulas,
    empirical_tail_dependence,
    tail_dependence_matrix,
    sample_gaussian_copula,
    sample_student_t_copula,
    sample_archimedean_copula,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_copula")

ARCHIMEDEAN = ["clayton", "gumbel", "frank"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fat-Tail Risk – Copula Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", "-c", default="configs/copula.yaml")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Danh sách ticker yfinance (vd: SPY TLT GLD)",
    )
    parser.add_argument("--csv", default=None,
                        help="CSV file với nhiều cột return (index=date)")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument(
        "--families", nargs="+",
        default=None,
        choices=["gaussian", "student_t", "clayton", "gumbel", "frank", "all"],
        help="Copula families cần fit (default: all)",
    )
    parser.add_argument(
        "--rank-by", choices=["aic", "bic", "loglik"], default="aic",
        help="Tiêu chí so sánh model (default: aic)",
    )
    parser.add_argument(
        "--tail-quantile", type=float, default=0.05,
        help="Ngưỡng đuôi cho tail dependence (default: 0.05)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10_000,
        help="Số mẫu sinh từ copula tốt nhất (default: 10_000)",
    )
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_multi_returns(args: argparse.Namespace, cfg: dict) -> pd.DataFrame:
    """Load returns của nhiều tài sản thành DataFrame."""
    if args.csv:
        logger.info(f"Loading from CSV: {args.csv}")
        df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        return df.dropna()

    tickers = (
        args.tickers
        or cfg.get("assets", {}).get("tickers", ["SPY", "TLT", "GLD"])
    )

    try:
        import yfinance as yf
        logger.info(f"Fetching tickers: {tickers} ({args.start} → {args.end})")
        raw = yf.download(tickers, start=args.start, end=args.end,
                          progress=False)["Adj Close"]
        returns = np.log(raw / raw.shift(1)).dropna()
        logger.info(f"Loaded {len(returns):,} observations × {len(tickers)} assets")
        return returns
    except Exception as e:
        logger.warning(f"yfinance failed ({e}). Using synthetic correlated returns.")

    # Synthetic: correlated Normal returns
    rng = np.random.default_rng(42)
    n_assets = max(len(args.tickers or []), 3)
    n_obs = 252 * 10
    corr = 0.4
    cov = np.full((n_assets, n_assets), corr)
    np.fill_diagonal(cov, 1.0)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_obs, n_assets)) @ L.T
    cols = args.tickers or [f"Asset_{i+1}" for i in range(n_assets)]
    returns = pd.DataFrame(z * 0.01, columns=cols[:n_assets])
    logger.info(f"Generated {n_obs} synthetic returns for {n_assets} assets")
    return returns


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_copula_figures(
    returns_df: pd.DataFrame,
    u: np.ndarray,
    best_family: str,
    best_result,
    output_dir: Path,
    tail_q: float,
) -> None:
    """Vẽ và lưu copula figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.visualization import plot_tail_dependence, plot_correlation

        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        cols = returns_df.columns.tolist()

        # Correlation heatmap
        fig = plot_correlation(returns_df, method="kendall",
                               title="Kendall's Tau Correlation")
        fig.savefig(fig_dir / "kendall_correlation.png",
                    bbox_inches="tight", dpi=120)
        plt.close(fig)

        # Tail dependence scatter (tất cả cặp)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a1, a2 = cols[i], cols[j]
                try:
                    fig = plot_tail_dependence(
                        returns_df, a1, a2,
                        quantile=tail_q,
                        title=f"Tail Dependence: {a1} vs {a2}",
                    )
                    fig.savefig(
                        fig_dir / f"tail_dep_{a1}_{a2}.png",
                        bbox_inches="tight", dpi=120,
                    )
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"  Tail dep scatter {a1}-{a2}: {e}")

        # Pseudo-obs scatter (bivariate)
        if u.shape[1] == 2:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(u[:, 0], u[:, 1], s=4, alpha=0.3,
                       color="#2B6CB0")
            ax.set_xlabel(f"u({cols[0]})")
            ax.set_ylabel(f"u({cols[1]})")
            ax.set_title(f"Pseudo-Observations – {cols[0]} vs {cols[1]}")
            fig.savefig(fig_dir / "pseudo_obs.png",
                        bbox_inches="tight", dpi=120)
            plt.close(fig)

        logger.info(f"Figures saved to {fig_dir}")

    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f"\n{'=' * 55}\n  {title}\n{'=' * 55}")


def print_copula_result(result) -> None:
    print(f"  Family         : {result.family}")
    for k, v in result.params.items():
        if k == "correlation_matrix":
            print(f"  Correlation    : (matrix, shape "
                  f"{len(v)}×{len(v[0])})")
        elif isinstance(v, float):
            print(f"  {k:<15}: {v:.4f}")
        else:
            print(f"  {k:<15}: {v}")
    print(f"  Log-likelihood : {result.log_lik:.2f}")
    print(f"  AIC            : {result.aic:.2f}")
    print(f"  BIC            : {result.bic:.2f}")
    print(f"  Tail dep λ_L   : {result.tail_dep_lower:.4f}")
    print(f"  Tail dep λ_U   : {result.tail_dep_upper:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    t0 = time.time()
    logger.info("=" * 55)
    logger.info("  Fat-Tail Risk – run_copula.py")
    logger.info("=" * 55)

    # ── 1. Config & output ───────────────────────────────────────────
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    output_dir = Path(
        args.output_dir
        or cfg.get("output", {}).get("dir", "outputs/copula")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    families = (
        args.families
        if args.families and "all" not in args.families
        else cfg.get("families", {}).get(
            "active", ["gaussian", "student_t", "clayton", "gumbel", "frank"]
        )
    )
    tail_q = (
        args.tail_quantile
        or cfg.get("tail_dependence", {}).get("quantiles", [0.05])[0]
    )

    logger.info(f"Families    : {families}")
    logger.info(f"Tail quant  : {tail_q}")
    logger.info(f"Output dir  : {output_dir}")

    # ── 2. Data ──────────────────────────────────────────────────────
    returns_df = load_multi_returns(args, cfg)
    cols = returns_df.columns.tolist()
    R = returns_df.values
    n, d = R.shape
    logger.info(f"Data shape  : {n:,} obs × {d} assets: {cols}")

    # ── 3. Pseudo-observations ───────────────────────────────────────
    print_section("Pseudo-Observations (Empirical CDF Transform)")
    u = pseudo_observations(R)
    print(f"  u shape: {u.shape}  |  range: [{u.min():.4f}, {u.max():.4f}]")

    # ── 4. Fit all copulas ───────────────────────────────────────────
    print_section("Copula Model Comparison")
    try:
        comparison_df = fit_all_copulas(u, rank_by=args.rank_by)
        cols_show = [c for c in ["log_lik", "aic", "bic",
                                  "tail_dep_lower", "tail_dep_upper"]
                     if c in comparison_df.columns]
        print(comparison_df[cols_show].round(4).to_string())

        # Save CSV
        path = output_dir / "copula_comparison.csv"
        comparison_df.to_csv(path)
        logger.info(f"Saved: {path}")

        best_family = comparison_df.index[0]
        logger.info(f"\nBest copula: {best_family} (by {args.rank_by.upper()})")

    except Exception as e:
        logger.error(f"Copula comparison failed: {e}")
        best_family = "gaussian"

    # ── 5. Fit best copula individually (detail) ─────────────────────
    print_section(f"Best Fit Detail: {best_family.upper()}")
    best_result = None
    try:
        if best_family == "gaussian":
            best_result = fit_gaussian_copula(u)
        elif best_family == "student_t":
            best_result = fit_student_t_copula(u)
        elif best_family in ARCHIMEDEAN:
            best_result = fit_archimedean_copula(u, family=best_family)
        if best_result:
            print_copula_result(best_result)
    except Exception as e:
        logger.warning(f"Detailed fit failed: {e}")

    # ── 6. Tail dependence matrix ────────────────────────────────────
    print_section(f"Empirical Tail Dependence (quantile={tail_q:.0%})")
    try:
        lower_df, upper_df = tail_dependence_matrix(R, quantile=tail_q)
        lower_df.index = upper_df.index = returns_df.columns
        lower_df.columns = upper_df.columns = returns_df.columns

        print(f"\n  Lower tail (λ_L):")
        print(lower_df.round(4).to_string())
        print(f"\n  Upper tail (λ_U):")
        print(upper_df.round(4).to_string())

        lower_df.to_csv(output_dir / "tail_dep_lower.csv")
        upper_df.to_csv(output_dir / "tail_dep_upper.csv")
        logger.info("Saved tail dependence matrices.")

    except Exception as e:
        logger.warning(f"Tail dependence failed: {e}")

    # ── 7. Bivariate pair analysis (nếu d == 2) ──────────────────────
    if d == 2:
        print_section(f"Bivariate Tail Dependence: {returns_df.columns[0]} – {returns_df.columns[1]}")
        try:
            lam_l, lam_u = empirical_tail_dependence(u, quantile=tail_q)
            print(f"  λ_lower = {lam_l:.4f}  (joint lower-tail co-movement)")
            print(f"  λ_upper = {lam_u:.4f}  (joint upper-tail co-movement)")
        except Exception as e:
            logger.warning(f"Bivariate tail dep failed: {e}")

    # ── 8. Sample from best copula ───────────────────────────────────
    print_section(f"Sampling {args.n_samples:,} observations from {best_family}")
    try:
        if best_result and best_family == "gaussian":
            import numpy as np
            R_corr = np.array(best_result.params["correlation_matrix"])
            samples = sample_gaussian_copula(args.n_samples, R_corr, seed=42)
        elif best_result and best_family == "student_t":
            df_val = best_result.params["df"]
            R_corr = np.array(best_result.params["correlation_matrix"])
            samples = sample_student_t_copula(args.n_samples, df_val, R_corr, seed=42)
        elif best_result and best_family in ARCHIMEDEAN and d == 2:
            theta = best_result.params["theta"]
            samples = sample_archimedean_copula(
                args.n_samples, theta=theta, family=best_family, seed=42
            )
        else:
            samples = None

        if samples is not None:
            print(f"  Sample shape : {samples.shape}")
            print(f"  u1 mean/std  : {samples[:, 0].mean():.3f} / {samples[:, 0].std():.3f}")
            if samples.shape[1] > 1:
                print(f"  u2 mean/std  : {samples[:, 1].mean():.3f} / {samples[:, 1].std():.3f}")

            path = output_dir / f"copula_samples_{best_family}.npy"
            np.save(path, samples)
            logger.info(f"Saved samples: {path}")

    except Exception as e:
        logger.warning(f"Sampling failed: {e}")

    # ── 9. Figures ───────────────────────────────────────────────────
    if not args.no_figures:
        save_copula_figures(
            returns_df, u, best_family, best_result,
            output_dir, tail_q,
        )

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.1f}s → {output_dir}")


if __name__ == "__main__":
    main()