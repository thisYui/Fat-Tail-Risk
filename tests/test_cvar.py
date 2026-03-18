"""
tests/test_cvar.py
-------------------
Unit tests cho CVaR / Expected Shortfall (ES).

Coverage
--------
  - Historical CVaR
  - Rolling historical CVaR
  - Parametric CVaR (Normal, Student-t, NIG)
  - GARCH CVaR
  - Filtered HS CVaR
  - CVaRResult dataclass
  - Component CVaR (Euler allocation)
  - compare_cvar_methods()
  - CVaR ≥ VaR property (coherence)
  - Sub-additivity check
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests import (
    make_normal_returns,
    make_fat_tail_returns,
    make_garch_returns,
    make_return_series,
    make_multi_returns,
    SEED, TOL, RTOL, N_OBS,
)
from src.models.cvar_models import (
    CVaRResult,
    historical_cvar,
    rolling_historical_cvar,
    parametric_cvar,
    garch_cvar,
    filtered_hs_cvar,
    cvar_contribution,
    compare_cvar_methods,
)
from src.models.var_models import hs_var, parametric_var


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_returns():
    return make_normal_returns(n=N_OBS, seed=SEED)


@pytest.fixture
def fat_tail_returns():
    return make_fat_tail_returns(n=N_OBS, df=4.0, seed=SEED)


@pytest.fixture
def garch_returns():
    return make_garch_returns(n=N_OBS, seed=SEED)


@pytest.fixture
def return_series():
    return make_return_series(n=N_OBS, seed=SEED)


@pytest.fixture
def multi_returns():
    return make_multi_returns(n=N_OBS, d=3, seed=SEED)


# ---------------------------------------------------------------------------
# CVaRResult dataclass
# ---------------------------------------------------------------------------

class TestCVaRResult:

    def test_creation(self):
        r = CVaRResult(cvar=-0.035, var=-0.025,
                       confidence=0.95, method="test")
        assert r.cvar == -0.035
        assert r.var  == -0.025
        assert r.confidence == 0.95

    def test_es_ratio(self):
        """CVaR / VaR ratio phải > 1 (CVaR âm hơn VaR)."""
        r = CVaRResult(cvar=-0.04, var=-0.025,
                       confidence=0.95, method="test")
        # es_ratio = cvar / var = -0.04 / -0.025 = 1.6
        assert r.es_ratio > 1.0

    def test_es_ratio_zero_var(self):
        r = CVaRResult(cvar=-0.01, var=0.0,
                       confidence=0.95, method="test")
        assert np.isnan(r.es_ratio)

    def test_as_series(self):
        r = CVaRResult(cvar=-0.04, var=-0.025,
                       confidence=0.95, method="test",
                       n_tail_obs=100)
        s = r.as_series()
        assert "cvar" in s.index
        assert "var" in s.index
        assert "es_ratio" in s.index
        assert "n_tail_obs" in s.index

    def test_cvar_is_negative(self, normal_returns):
        result = historical_cvar(normal_returns, 0.95)
        assert result.cvar < 0


# ---------------------------------------------------------------------------
# Historical CVaR
# ---------------------------------------------------------------------------

class TestHistoricalCVaR:

    def test_basic(self, normal_returns):
        result = historical_cvar(normal_returns, 0.95)
        assert isinstance(result, CVaRResult)
        assert result.cvar < 0
        assert result.method == "historical"

    def test_cvar_more_negative_than_var(self, normal_returns):
        """CVaR phải âm hơn hoặc bằng VaR – tính coherence cơ bản."""
        result = historical_cvar(normal_returns, 0.95)
        assert result.cvar <= result.var + TOL, \
            f"CVaR ({result.cvar:.4f}) should be ≤ VaR ({result.var:.4f})"

    def test_confidence_monotonicity(self, normal_returns):
        """CVaR 99% phải âm hơn CVaR 95%."""
        cvar95 = historical_cvar(normal_returns, 0.95).cvar
        cvar99 = historical_cvar(normal_returns, 0.99).cvar
        assert cvar99 <= cvar95, "Higher confidence → more negative CVaR"

    def test_tail_obs_positive(self, normal_returns):
        result = historical_cvar(normal_returns, 0.95)
        assert result.n_tail_obs > 0
        expected_tail = int(N_OBS * 0.05)
        assert abs(result.n_tail_obs - expected_tail) <= expected_tail * 0.2

    def test_cvar_is_mean_of_tail(self, normal_returns):
        """CVaR = E[R | R ≤ VaR] – kiểm tra trực tiếp."""
        confidence = 0.95
        result  = historical_cvar(normal_returns, confidence)
        var_val = result.var
        tail    = normal_returns[normal_returns <= var_val]
        expected_cvar = float(tail.mean())
        assert abs(result.cvar - expected_cvar) < TOL

    def test_fat_tail_more_negative(self, normal_returns, fat_tail_returns):
        """Fat-tail CVaR 99% phải âm hơn Normal."""
        cvar_n  = historical_cvar(normal_returns,   0.99).cvar
        cvar_ft = historical_cvar(fat_tail_returns, 0.99).cvar
        assert cvar_ft <= cvar_n

    def test_window_parameter(self, normal_returns):
        result = historical_cvar(normal_returns, 0.95, window=500)
        assert result.cvar < 0
        assert result.params.get("n_obs") == 500

    def test_nan_handling(self):
        r = make_normal_returns(n=300, seed=SEED).astype(float)
        r[::15] = np.nan
        result = historical_cvar(r, 0.95)
        assert np.isfinite(result.cvar)


# ---------------------------------------------------------------------------
# Rolling Historical CVaR
# ---------------------------------------------------------------------------

class TestRollingHistoricalCVaR:

    def test_returns_series(self, return_series):
        rv = rolling_historical_cvar(return_series, 0.95, window=252)
        assert len(rv) == len(return_series)
        assert rv.iloc[:251].isna().all()
        assert (rv.dropna() < 0).all()

    def test_cvar_leq_var_rolling(self, return_series):
        """Rolling CVaR ≤ rolling VaR at every point."""
        window = 252
        rv_var  = return_series.rolling(window).quantile(0.05)
        rv_cvar = rolling_historical_cvar(return_series, 0.95, window)
        valid   = rv_cvar.dropna().index
        assert (rv_cvar[valid] <= rv_var[valid] + TOL).all()

    def test_monotonicity_across_confidence(self, return_series):
        rv95 = rolling_historical_cvar(return_series, 0.95, 252).dropna()
        rv99 = rolling_historical_cvar(return_series, 0.99, 252).dropna()
        assert (rv99 <= rv95 + TOL).all()


# ---------------------------------------------------------------------------
# Parametric CVaR
# ---------------------------------------------------------------------------

class TestParametricCVaR:

    @pytest.mark.parametrize("dist", ["normal", "student_t", "nig"])
    def test_basic(self, normal_returns, dist):
        result = parametric_cvar(normal_returns, 0.95, dist)
        assert isinstance(result, CVaRResult)
        assert result.cvar < 0
        assert dist in result.method

    def test_cvar_leq_var(self, normal_returns):
        for dist in ["normal", "student_t"]:
            result = parametric_cvar(normal_returns, 0.95, dist)
            assert result.cvar <= result.var + TOL, \
                f"{dist}: CVaR should be ≤ VaR"

    def test_normal_closed_form(self, normal_returns):
        """
        Normal CVaR = μ - σ·φ(z_α)/α.
        Kiểm tra xấp xỉ so với historical.
        """
        r_param = parametric_cvar(normal_returns, 0.95, "normal").cvar
        r_hist  = historical_cvar(normal_returns, 0.95).cvar
        # Cho phép sai số 15% relative
        assert abs(r_param - r_hist) / abs(r_hist) < 0.15

    def test_student_t_fatter_than_normal(self, fat_tail_returns):
        """Student-t CVaR 99% phải âm hơn Normal CVaR."""
        cvar_n = parametric_cvar(fat_tail_returns, 0.99, "normal").cvar
        cvar_t = parametric_cvar(fat_tail_returns, 0.99, "student_t").cvar
        assert cvar_t <= cvar_n

    def test_confidence_monotonicity(self, normal_returns):
        for dist in ["normal", "student_t"]:
            c95 = parametric_cvar(normal_returns, 0.95, dist).cvar
            c99 = parametric_cvar(normal_returns, 0.99, dist).cvar
            assert c99 <= c95, f"{dist}: CVaR99 should ≤ CVaR95"

    def test_invalid_dist_raises(self, normal_returns):
        with pytest.raises(ValueError):
            parametric_cvar(normal_returns, 0.95, "invalid_dist")

    def test_nig_finite(self, normal_returns):
        result = parametric_cvar(normal_returns, 0.95, "nig")
        assert np.isfinite(result.cvar)
        assert np.isfinite(result.var)


# ---------------------------------------------------------------------------
# GARCH CVaR
# ---------------------------------------------------------------------------

class TestGARCHCVaR:

    @pytest.mark.parametrize("dist", ["normal", "student_t"])
    def test_basic(self, garch_returns, dist):
        result = garch_cvar(garch_returns, 0.95, dist=dist)
        assert isinstance(result, CVaRResult)
        assert result.cvar < 0
        assert dist in result.method

    def test_cvar_leq_var(self, garch_returns):
        for dist in ["normal", "student_t"]:
            result = garch_cvar(garch_returns, 0.95, dist=dist)
            assert result.cvar <= result.var + TOL

    def test_sigma_next_stored(self, garch_returns):
        result = garch_cvar(garch_returns, 0.95)
        assert "sigma_next" in result.params
        assert result.params["sigma_next"] > 0

    def test_invalid_dist_raises(self, garch_returns):
        with pytest.raises(ValueError):
            garch_cvar(garch_returns, 0.95, dist="bad_dist")


# ---------------------------------------------------------------------------
# Filtered HS CVaR
# ---------------------------------------------------------------------------

class TestFilteredHSCVaR:

    def test_basic(self, garch_returns):
        result = filtered_hs_cvar(garch_returns, 0.95)
        assert isinstance(result, CVaRResult)
        assert result.cvar < 0
        assert result.method == "filtered_hs"

    def test_cvar_leq_var(self, garch_returns):
        result = filtered_hs_cvar(garch_returns, 0.95)
        assert result.cvar <= result.var + TOL

    def test_n_tail_obs(self, garch_returns):
        result = filtered_hs_cvar(garch_returns, 0.95)
        assert result.n_tail_obs > 0

    def test_confidence_monotonicity(self, garch_returns):
        c95 = filtered_hs_cvar(garch_returns, 0.95).cvar
        c99 = filtered_hs_cvar(garch_returns, 0.99).cvar
        assert c99 <= c95


# ---------------------------------------------------------------------------
# Component CVaR
# ---------------------------------------------------------------------------

class TestComponentCVaR:

    def test_basic(self, multi_returns):
        weights = np.array([0.5, 0.3, 0.2])
        comp = cvar_contribution(weights, multi_returns, 0.95)
        assert len(comp) == 3
        assert comp.sum() != 0  # không phải tất cả zero

    def test_sum_equals_portfolio_cvar(self, multi_returns):
        """
        Tổng Component CVaR ≈ Portfolio CVaR (Euler allocation).
        """
        weights = np.array([0.5, 0.3, 0.2])
        w = weights / weights.sum()

        port_ret = multi_returns @ w
        port_cvar = historical_cvar(port_ret, 0.95).cvar
        comp = cvar_contribution(w, multi_returns, 0.95)

        assert abs(comp.sum() - port_cvar) / abs(port_cvar) < 0.10

    def test_equal_weights(self, multi_returns):
        d = multi_returns.shape[1]
        w = np.ones(d) / d
        comp = cvar_contribution(w, multi_returns, 0.95)
        assert len(comp) == d
        # Tổng không được bằng 0
        assert abs(comp.sum()) > TOL

    def test_single_asset(self):
        """Với 1 tài sản, component CVaR = portfolio CVaR."""
        r = make_normal_returns(n=N_OBS, seed=SEED).reshape(-1, 1)
        w = np.array([1.0])
        comp = cvar_contribution(w, r, 0.95)
        port_cvar = historical_cvar(r[:, 0], 0.95).cvar
        assert abs(comp[0] - port_cvar) / abs(port_cvar) < 0.01


# ---------------------------------------------------------------------------
# compare_cvar_methods()
# ---------------------------------------------------------------------------

class TestCompareCVaRMethods:

    def test_returns_dataframe(self, normal_returns):
        df = compare_cvar_methods(normal_returns, 0.95)
        assert hasattr(df, "index")
        assert "cvar" in df.columns
        assert "var" in df.columns

    def test_all_methods_present(self, normal_returns):
        df = compare_cvar_methods(normal_returns, 0.95)
        expected = {"historical", "parametric_normal", "parametric_t",
                    "garch_normal", "garch_t", "filtered_hs"}
        assert expected.issubset(set(df.index))

    def test_all_cvar_more_negative_than_var(self, normal_returns):
        df = compare_cvar_methods(normal_returns, 0.95).dropna()
        valid = df.dropna(subset=["cvar", "var"])
        assert (valid["cvar"] <= valid["var"] + TOL).all(), \
            "All CVaR ≤ VaR"

    def test_sorted_ascending(self, normal_returns):
        df = compare_cvar_methods(normal_returns, 0.95).dropna()
        assert df["cvar"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Coherence & ordering properties
# ---------------------------------------------------------------------------

class TestCVaRCoherence:

    def test_cvar_geq_var_always(self, normal_returns, fat_tail_returns):
        """CVaR ≤ VaR (âm hơn) – property cơ bản của ES."""
        for r in [normal_returns, fat_tail_returns]:
            for cl in [0.90, 0.95, 0.99]:
                cvar = historical_cvar(r, cl).cvar
                var  = hs_var(r, cl).var
                assert cvar <= var + TOL, \
                    f"CVaR {cl:.0%} ({cvar:.4f}) should be ≤ VaR ({var:.4f})"

    def test_cvar_not_equal_to_var(self, normal_returns):
        """CVaR và VaR không nên bằng nhau với phân phối liên tục."""
        result = historical_cvar(normal_returns, 0.95)
        assert result.cvar < result.var - TOL, \
            "CVaR should be strictly more negative than VaR"

    def test_cvar_bounded_by_min(self, normal_returns):
        """CVaR không thể âm hơn giá trị nhỏ nhất."""
        result = historical_cvar(normal_returns, 0.99)
        assert result.cvar >= normal_returns.min() - TOL

    def test_cvar_is_finite(self, normal_returns, fat_tail_returns):
        for r in [normal_returns, fat_tail_returns]:
            result = historical_cvar(r, 0.99)
            assert np.isfinite(result.cvar)

    def test_larger_sigma_more_negative_cvar(self):
        """Volatility cao hơn → CVaR âm hơn."""
        r_low  = make_normal_returns(n=N_OBS, sigma=0.005, seed=SEED)
        r_high = make_normal_returns(n=N_OBS, sigma=0.020, seed=SEED)
        cvar_low  = historical_cvar(r_low,  0.95).cvar
        cvar_high = historical_cvar(r_high, 0.95).cvar
        assert cvar_high <= cvar_low


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestCVaREdgeCases:

    def test_all_same_returns(self):
        r = np.full(500, -0.01)
        result = historical_cvar(r, 0.95)
        assert abs(result.cvar - (-0.01)) < TOL

    def test_single_tail_obs(self):
        """Khi chỉ có 1 quan sát trong đuôi."""
        r = np.zeros(1000)
        r[0] = -1.0   # chỉ 1 giá trị cực âm
        result = historical_cvar(r, 0.999)
        assert np.isfinite(result.cvar)

    def test_very_short_series(self):
        r = make_normal_returns(n=30, seed=SEED)
        result = historical_cvar(r, 0.90)
        assert result.cvar < 0

    def test_large_n(self):
        r = make_normal_returns(n=100_000, seed=SEED)
        result = historical_cvar(r, 0.99)
        assert result.cvar < 0
        assert np.isfinite(result.cvar)

    def test_reproducibility(self):
        r1 = make_normal_returns(n=500, seed=42)
        r2 = make_normal_returns(n=500, seed=42)
        c1 = historical_cvar(r1, 0.95).cvar
        c2 = historical_cvar(r2, 0.95).cvar
        assert c1 == c2