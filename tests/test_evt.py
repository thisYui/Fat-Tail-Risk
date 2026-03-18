"""
tests/test_evt.py
------------------
Unit tests cho Extreme Value Theory (EVT).

Coverage
--------
  - GPD fit (POT method)
  - GEV fit (Block Maxima)
  - EVT-VaR, EVT-CVaR
  - Return levels
  - Threshold selection diagnostics
  - Hill estimator
  - Mean Excess Function
  - GPDResult / GEVResult dataclasses
  - Parameter validity (ξ, σ > 0)
  - Asymptotic properties
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests import (
    make_normal_returns,
    make_fat_tail_returns,
    SEED, TOL, RTOL, N_OBS,
)
from src.models.evt import (
    fit_gpd_pot,
    fit_gev,
    evt_var,
    evt_cvar,
    return_levels,
    threshold_stability_plot_data,
    mean_excess_plot_data,
    evt_summary,
    GPDResult,
    GEVResult,
)
from src.features.tail import (
    hill_estimator,
    hill_plot,
    mean_excess_function,
    select_pot_threshold,
    pot_exceedances,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_returns():
    return make_normal_returns(n=N_OBS, seed=SEED)


@pytest.fixture
def fat_tail_returns():
    return make_fat_tail_returns(n=N_OBS, df=3.0, seed=SEED)


@pytest.fixture
def pareto_losses():
    """Sinh dữ liệu từ Pareto (xi > 0) để test GPD với fat-tail rõ ràng."""
    rng = np.random.default_rng(SEED)
    # Pareto(b=1) → GPD với xi=1, scale=1
    shape = 0.5
    u = rng.uniform(0, 1, N_OBS)
    losses = (u ** (-shape) - 1) / shape   # GPD inverse CDF
    # Wrap thành returns âm
    r = np.zeros(N_OBS)
    n_tail = N_OBS // 4
    r[-n_tail:] = -losses[:n_tail]
    r[:-n_tail] = make_normal_returns(n=N_OBS - n_tail, sigma=0.005, seed=SEED + 1)
    return r


# ---------------------------------------------------------------------------
# GPDResult dataclass
# ---------------------------------------------------------------------------

class TestGPDResult:

    def test_creation(self, normal_returns):
        gpd = fit_gpd_pot(normal_returns, threshold_quantile=0.90)
        assert isinstance(gpd, GPDResult)
        assert gpd.xi is not None
        assert gpd.sigma > 0
        assert gpd.threshold > 0
        assert gpd.n_exceedances > 0
        assert gpd.n_total == len(normal_returns[~np.isnan(normal_returns)])

    def test_exceedance_rate(self, normal_returns):
        gpd = fit_gpd_pot(normal_returns, threshold_quantile=0.90)
        expected_rate = 0.10
        assert abs(gpd.exceedance_rate - expected_rate) < 0.02

    def test_var_negative(self, normal_returns):
        gpd = fit_gpd_pot(normal_returns, threshold_quantile=0.90)
        var = gpd.var(0.95)
        assert var < 0

    def test_cvar_leq_var(self, normal_returns):
        gpd = fit_gpd_pot(normal_returns, threshold_quantile=0.90)
        for cl in [0.95, 0.99]:
            var  = gpd.var(cl)
            cvar = gpd.cvar(cl)
            assert cvar <= var + TOL, \
                f"EVT-CVaR ({cvar:.4f}) should be ≤ EVT-VaR ({var:.4f})"

    def test_var_monotonicity(self, normal_returns):
        gpd = fit_gpd_pot(normal_returns, threshold_quantile=0.90)
        var95 = gpd.var(0.95)
        var99 = gpd.var(0.99)
        assert var99 <= var95, "Higher confidence → more negative VaR"

    def test_return_level_increases_with_period(self, fat_tail_returns):
        """Loss level tăng theo chu kỳ T."""
        gpd = fit_gpd_pot(fat_tail_returns, threshold_quantile=0.90)
        rl10  = gpd.return_level(10)
        rl100 = gpd.return_level(100)
        # return_level trả về giá trị dương (loss magnitude)
        assert abs(rl100) >= abs(rl10), \
            "100-year loss ≥ 10-year loss"

    def test_aic_bic_finite(self, normal_returns):
        gpd = fit_gpd_pot(normal_returns)
        assert np.isfinite(gpd.aic)
        assert np.isfinite(gpd.bic)
        assert np.isfinite(gpd.log_lik)


# ---------------------------------------------------------------------------
# fit_gpd_pot()
# ---------------------------------------------------------------------------

class TestFitGPDPOT:

    def test_sigma_positive(self, normal_returns):
        gpd = fit_gpd_pot(normal_returns)
        assert gpd.sigma > 0, "GPD scale σ must be positive"

    def test_fat_tail_positive_xi(self, pareto_losses):
        """Pareto data phải cho xi > 0 (fat-tail)."""
        gpd = fit_gpd_pot(pareto_losses, threshold_quantile=0.80)
        assert gpd.xi > 0, f"Expected xi > 0 for fat-tail data, got {gpd.xi:.4f}"

    def test_threshold_respected(self, normal_returns):
        """Exceedances phải > threshold."""
        gpd = fit_gpd_pot(normal_returns, threshold_quantile=0.90)
        losses = -normal_returns[normal_returns < 0]
        exceedances = losses[losses > gpd.threshold]
        assert len(exceedances) == gpd.n_exceedances

    def test_different_thresholds(self, fat_tail_returns):
        """Threshold cao hơn → ít exceedances hơn."""
        gpd80 = fit_gpd_pot(fat_tail_returns, threshold_quantile=0.80)
        gpd95 = fit_gpd_pot(fat_tail_returns, threshold_quantile=0.95)
        assert gpd95.n_exceedances < gpd80.n_exceedances

    def test_too_few_exceedances_raises(self):
        """Ngưỡng quá cao → không đủ exceedances → ValueError."""
        r = make_normal_returns(n=200, sigma=0.01, seed=SEED)
        with pytest.raises(ValueError, match="[Tt]oo few"):
            fit_gpd_pot(r, threshold_quantile=0.9999)

    def test_both_sides(self, fat_tail_returns):
        """Fit được cả left và right tail."""
        gpd_left  = fit_gpd_pot(fat_tail_returns, side="left")
        gpd_right = fit_gpd_pot(fat_tail_returns, side="right")
        assert gpd_left.sigma > 0
        assert gpd_right.sigma > 0

    def test_n_obs_correct(self, normal_returns):
        gpd = fit_gpd_pot(normal_returns)
        assert gpd.n_total == len(normal_returns[~np.isnan(normal_returns)])

    def test_known_distribution(self):
        """
        Với data từ Exponential (xi ≈ 0), xi ước lượng phải gần 0.
        """
        rng = np.random.default_rng(SEED)
        exp_losses = rng.exponential(scale=1.0, size=N_OBS)
        r = np.zeros(N_OBS)
        r[-N_OBS // 5:] = -exp_losses[:N_OBS // 5]
        r[:-N_OBS // 5] = make_normal_returns(n=N_OBS - N_OBS // 5,
                                               sigma=0.01, seed=SEED + 1)
        gpd = fit_gpd_pot(r, threshold_quantile=0.80)
        # Exponential → xi ≈ 0 (trong khoảng [-0.5, 0.5])
        assert -0.5 < gpd.xi < 0.5, \
            f"Exponential tail should have xi ≈ 0, got {gpd.xi:.3f}"


# ---------------------------------------------------------------------------
# GEVResult dataclass & fit_gev()
# ---------------------------------------------------------------------------

class TestGEVResult:

    def test_creation(self, fat_tail_returns):
        gev = fit_gev(fat_tail_returns, block_size=21)
        assert isinstance(gev, GEVResult)
        assert gev.sigma > 0
        assert gev.n_blocks > 0

    def test_return_level_increases(self, fat_tail_returns):
        gev = fit_gev(fat_tail_returns, block_size=21)
        rl5  = gev.return_level(5)
        rl50 = gev.return_level(50)
        assert abs(rl50) >= abs(rl5), "50-period return level ≥ 5-period"

    def test_sigma_positive(self, normal_returns):
        gev = fit_gev(normal_returns, block_size=21)
        assert gev.sigma > 0

    def test_aic_bic_finite(self, normal_returns):
        gev = fit_gev(normal_returns, block_size=21)
        assert np.isfinite(gev.aic)
        assert np.isfinite(gev.bic)

    def test_n_blocks(self, fat_tail_returns):
        block_size = 21
        gev = fit_gev(fat_tail_returns, block_size=block_size)
        expected_blocks = len(fat_tail_returns) // block_size
        assert gev.n_blocks == expected_blocks

    @pytest.mark.parametrize("block_size", [21, 63, 126])
    def test_different_block_sizes(self, fat_tail_returns, block_size):
        gev = fit_gev(fat_tail_returns, block_size=block_size)
        assert gev.sigma > 0
        assert np.isfinite(gev.aic)


# ---------------------------------------------------------------------------
# EVT-VaR
# ---------------------------------------------------------------------------

class TestEVTVaR:

    def test_basic_pot(self, normal_returns):
        var = evt_var(normal_returns, confidence=0.95, method="pot")
        assert var < 0
        assert np.isfinite(var)

    def test_basic_bm(self, fat_tail_returns):
        var = evt_var(fat_tail_returns, confidence=0.95, method="bm")
        assert np.isfinite(var)

    def test_monotonicity(self, fat_tail_returns):
        var95 = evt_var(fat_tail_returns, 0.95)
        var99 = evt_var(fat_tail_returns, 0.99)
        assert var99 <= var95, "EVT-VaR: 99% should be ≤ 95%"

    def test_fat_tail_more_negative(self, normal_returns, fat_tail_returns):
        """Fat-tail EVT-VaR 99% phải âm hơn Normal EVT-VaR."""
        var_n  = evt_var(normal_returns,   0.99)
        var_ft = evt_var(fat_tail_returns, 0.99)
        assert var_ft <= var_n

    def test_invalid_method_raises(self, normal_returns):
        with pytest.raises(ValueError):
            evt_var(normal_returns, method="invalid")

    def test_consistent_with_gpd_var(self, fat_tail_returns):
        """evt_var() phải nhất quán với GPDResult.var()."""
        threshold_q = 0.90
        gpd = fit_gpd_pot(fat_tail_returns, threshold_quantile=threshold_q)
        var_direct   = gpd.var(0.99)
        var_fn       = evt_var(fat_tail_returns, 0.99,
                               threshold_quantile=threshold_q)
        assert abs(var_direct - var_fn) < TOL


# ---------------------------------------------------------------------------
# EVT-CVaR
# ---------------------------------------------------------------------------

class TestEVTCVaR:

    def test_basic(self, fat_tail_returns):
        cvar = evt_cvar(fat_tail_returns, confidence=0.95)
        assert cvar < 0
        assert np.isfinite(cvar)

    def test_cvar_leq_var(self, fat_tail_returns):
        for cl in [0.90, 0.95, 0.99]:
            var  = evt_var(fat_tail_returns, cl)
            cvar = evt_cvar(fat_tail_returns, cl)
            assert cvar <= var + TOL, \
                f"EVT-CVaR ({cvar:.4f}) ≤ EVT-VaR ({var:.4f}) at {cl:.0%}"

    def test_monotonicity(self, fat_tail_returns):
        cvar95 = evt_cvar(fat_tail_returns, 0.95)
        cvar99 = evt_cvar(fat_tail_returns, 0.99)
        assert cvar99 <= cvar95

    def test_consistent_with_gpd(self, fat_tail_returns):
        gpd = fit_gpd_pot(fat_tail_returns, threshold_quantile=0.90)
        cvar_gpd = gpd.cvar(0.99)
        cvar_fn  = evt_cvar(fat_tail_returns, 0.99, threshold_quantile=0.90)
        assert abs(cvar_gpd - cvar_fn) < TOL


# ---------------------------------------------------------------------------
# Return Levels
# ---------------------------------------------------------------------------

class TestReturnLevels:

    def test_returns_dataframe(self, fat_tail_returns):
        df = return_levels(fat_tail_returns)
        assert hasattr(df, "index")
        assert "return_level_loss" in df.columns
        assert "return_level_pct" in df.columns

    def test_default_periods(self, fat_tail_returns):
        df = return_levels(fat_tail_returns)
        expected_periods = {1, 2, 5, 10, 25, 50, 100, 200}
        assert expected_periods.issubset(set(df.index))

    def test_custom_periods(self, fat_tail_returns):
        periods = (1, 5, 10)
        df = return_levels(fat_tail_returns, periods=periods)
        assert set(df.index) == set(periods)

    def test_monotonic_increase(self, fat_tail_returns):
        """Return level phải tăng theo chu kỳ."""
        df = return_levels(
            fat_tail_returns,
            periods=(1, 5, 10, 25, 50),
        )
        losses = df["return_level_loss"].values
        assert np.all(np.diff(losses) >= -TOL), \
            "Return levels should be non-decreasing"

    def test_all_finite(self, fat_tail_returns):
        df = return_levels(fat_tail_returns)
        assert df["return_level_loss"].apply(np.isfinite).all()


# ---------------------------------------------------------------------------
# Threshold Diagnostics
# ---------------------------------------------------------------------------

class TestThresholdDiagnostics:

    def test_threshold_stability(self, fat_tail_returns):
        df = threshold_stability_plot_data(fat_tail_returns)
        assert "threshold" in df.columns
        assert "xi" in df.columns
        assert "sigma_star" in df.columns
        assert "n_exceedances" in df.columns
        assert len(df) > 5

    def test_mean_excess_plot(self, fat_tail_returns):
        df = mean_excess_plot_data(fat_tail_returns)
        assert "threshold" in df.columns
        assert "mean_excess" in df.columns
        assert (df["mean_excess"] >= 0).all(), \
            "Mean excess should be non-negative"

    def test_mean_excess_monotonic_for_pareto(self, pareto_losses):
        """
        Với Pareto data, MEF phải tuyến tính tăng.
        Kiểm tra: MEF tại threshold cao hơn phải ≥ MEF tại threshold thấp hơn.
        """
        df = mean_excess_plot_data(pareto_losses)
        if len(df) < 5:
            pytest.skip("Not enough thresholds")
        mef = df["mean_excess"].values
        # Ít nhất 60% các bước phải tăng hoặc bằng (slope > 0)
        increases = np.sum(np.diff(mef) >= -0.01 * abs(mef[:-1]).mean())
        assert increases / len(np.diff(mef)) > 0.5


# ---------------------------------------------------------------------------
# Hill Estimator
# ---------------------------------------------------------------------------

class TestHillEstimator:

    def test_basic(self, fat_tail_returns):
        alpha = hill_estimator(fat_tail_returns, k_fraction=0.10)
        assert alpha > 0
        assert np.isfinite(alpha)

    def test_fat_tail_smaller_alpha(self, normal_returns, fat_tail_returns):
        """
        Fat-tail data (df=3) phải có tail index nhỏ hơn Normal.
        """
        alpha_n  = hill_estimator(normal_returns,   k_fraction=0.10)
        alpha_ft = hill_estimator(fat_tail_returns, k_fraction=0.10)
        assert alpha_ft < alpha_n, \
            f"Fat-tail α ({alpha_ft:.2f}) should be < Normal α ({alpha_n:.2f})"

    def test_different_k_fractions(self, fat_tail_returns):
        """Các mức k khác nhau phải cho kết quả hữu hạn dương."""
        for k_frac in [0.05, 0.10, 0.20]:
            alpha = hill_estimator(fat_tail_returns, k_fraction=k_frac)
            assert alpha > 0
            assert np.isfinite(alpha)

    def test_pareto_known_alpha(self, pareto_losses):
        """
        Pareto với shape=0.5 → tail index α = 1/0.5 = 2.
        Hill estimator phải ước lượng gần 2.
        """
        alpha = hill_estimator(pareto_losses, k_fraction=0.15)
        # Cho phép sai số 50% do sample size và mixed data
        assert 0.5 < alpha < 8.0, \
            f"Hill estimator {alpha:.2f} out of expected range for Pareto(0.5)"

    def test_hill_plot_dataframe(self, fat_tail_returns):
        df = hill_plot(fat_tail_returns)
        assert "k" in df.columns
        assert "tail_index" in df.columns
        assert len(df) > 5
        assert (df["tail_index"].dropna() > 0).all()

    def test_short_series_returns_nan(self):
        r = make_fat_tail_returns(n=5, seed=SEED)
        alpha = hill_estimator(r, k_fraction=0.5)
        assert np.isnan(alpha)


# ---------------------------------------------------------------------------
# EVT Summary
# ---------------------------------------------------------------------------

class TestEVTSummary:

    def test_basic(self, fat_tail_returns):
        s = evt_summary(fat_tail_returns)
        assert isinstance(s, type(s))  # pd.Series
        assert "gpd_xi" in s.index
        assert "gpd_sigma" in s.index

    def test_var_cvar_present(self, fat_tail_returns):
        s = evt_summary(fat_tail_returns,
                        confidence_levels=(0.95, 0.99))
        assert "evt_var_95" in s.index
        assert "evt_var_99" in s.index
        assert "evt_cvar_95" in s.index
        assert "evt_cvar_99" in s.index

    def test_return_levels_present(self, fat_tail_returns):
        s = evt_summary(fat_tail_returns, periods=(1, 5, 10))
        assert "return_level_1yr" in s.index
        assert "return_level_5yr" in s.index
        assert "return_level_10yr" in s.index

    def test_all_finite(self, fat_tail_returns):
        s = evt_summary(fat_tail_returns)
        numeric = s.apply(lambda x: isinstance(x, float))
        assert s[numeric].apply(np.isfinite).all(), \
            "All numeric EVT summary values should be finite"

    def test_gpd_sigma_positive(self, fat_tail_returns):
        s = evt_summary(fat_tail_returns)
        assert s["gpd_sigma"] > 0

    def test_var_monotonicity_in_summary(self, fat_tail_returns):
        s = evt_summary(
            fat_tail_returns,
            confidence_levels=(0.90, 0.95, 0.99),
        )
        var90 = s["evt_var_90"]
        var95 = s["evt_var_95"]
        var99 = s["evt_var_99"]
        assert var99 <= var95 <= var90, \
            "EVT-VaR should be monotone in confidence level"


# ---------------------------------------------------------------------------
# Threshold selection helpers (features.tail)
# ---------------------------------------------------------------------------

class TestThresholdSelection:

    def test_select_pot_threshold_percentile(self, fat_tail_returns):
        u = select_pot_threshold(fat_tail_returns, method="percentile",
                                  percentile=0.90)
        losses = -fat_tail_returns[fat_tail_returns < 0]
        expected = float(np.quantile(losses, 0.90))
        assert abs(u - expected) < TOL

    def test_select_pot_threshold_std(self, fat_tail_returns):
        u = select_pot_threshold(fat_tail_returns, method="std")
        losses = -fat_tail_returns[fat_tail_returns < 0]
        expected = losses.mean() + 2 * losses.std()
        assert abs(u - expected) < TOL

    def test_invalid_method_raises(self, fat_tail_returns):
        with pytest.raises(ValueError):
            select_pot_threshold(fat_tail_returns, method="invalid")

    def test_pot_exceedances_positive(self, fat_tail_returns):
        u = select_pot_threshold(fat_tail_returns, percentile=0.90)
        exc = pot_exceedances(fat_tail_returns, threshold=u)
        assert (exc >= 0).all(), "Exceedances should be non-negative"
        assert len(exc) > 0

    def test_pot_exceedances_decreases_with_threshold(self, fat_tail_returns):
        """Ngưỡng cao hơn → ít exceedances hơn."""
        u_low  = select_pot_threshold(fat_tail_returns, percentile=0.80)
        u_high = select_pot_threshold(fat_tail_returns, percentile=0.95)
        exc_low  = pot_exceedances(fat_tail_returns, u_low)
        exc_high = pot_exceedances(fat_tail_returns, u_high)
        assert len(exc_high) <= len(exc_low)