"""
tests/test_var.py
------------------
Unit tests cho tất cả các phương pháp Value-at-Risk (VaR).

Coverage
--------
  - Historical Simulation VaR
  - Parametric VaR (Normal, Student-t, NIG)
  - Cornish-Fisher VaR
  - GARCH VaR
  - Filtered Historical Simulation VaR
  - VaRResult dataclass
  - Rolling HS VaR
  - compare_var_methods()
  - Monotonicity, sign, ordering properties
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
    SEED, TOL, RTOL, N_OBS,
)
from src.models.var_models import (
    hs_var,
    rolling_hs_var,
    parametric_var,
    cornish_fisher_var,
    garch_var,
    filtered_hs_var,
    compare_var_methods,
    VaRResult,
)


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


# ---------------------------------------------------------------------------
# VaRResult dataclass
# ---------------------------------------------------------------------------

class TestVaRResult:

    def test_creation(self):
        r = VaRResult(var=-0.025, confidence=0.95, method="test")
        assert r.var == -0.025
        assert r.confidence == 0.95
        assert r.method == "test"
        assert r.horizon == 1
        assert isinstance(r.params, dict)

    def test_scale_to_horizon(self):
        r = VaRResult(var=-0.01, confidence=0.95, method="hs")
        r10 = r.scale_to_horizon(10)
        assert abs(r10.var - (-0.01 * np.sqrt(10))) < TOL
        assert r10.horizon == 10
        assert "scaled_10d" in r10.method

    def test_as_series(self):
        r = VaRResult(var=-0.02, confidence=0.99, method="test",
                      params={"foo": 1.0})
        s = r.as_series()
        assert "var" in s.index
        assert "confidence" in s.index
        assert "foo" in s.index

    def test_var_is_negative(self, normal_returns):
        """VaR loss-side phải là giá trị âm."""
        result = hs_var(normal_returns, confidence=0.95)
        assert result.var < 0, "VaR (loss side) should be negative"


# ---------------------------------------------------------------------------
# Historical Simulation VaR
# ---------------------------------------------------------------------------

class TestHistoricalSimulationVaR:

    def test_basic(self, normal_returns):
        result = hs_var(normal_returns, confidence=0.95)
        assert isinstance(result, VaRResult)
        assert result.var < 0
        assert result.method == "historical_simulation"

    def test_confidence_monotonicity(self, normal_returns):
        """VaR 99% phải tệ hơn (âm hơn) VaR 95%."""
        var95 = hs_var(normal_returns, 0.95).var
        var99 = hs_var(normal_returns, 0.99).var
        assert var99 <= var95, "Higher confidence → more negative VaR"

    def test_empirical_rate(self, normal_returns):
        """Tỷ lệ vi phạm thực tế phải gần với (1 - confidence)."""
        confidence = 0.95
        result = hs_var(normal_returns, confidence)
        viol_rate = np.mean(normal_returns < result.var)
        # Cho phép sai lệch ±2% so với lý thuyết
        assert abs(viol_rate - (1 - confidence)) < 0.02

    def test_window_parameter(self, normal_returns):
        """VaR với window nhỏ hơn chỉ dùng data gần đây."""
        result_full   = hs_var(normal_returns, 0.95, window=None)
        result_window = hs_var(normal_returns, 0.95, window=252)
        # Không nhất thiết bằng nhau, nhưng cả hai phải hợp lệ
        assert result_full.var < 0
        assert result_window.var < 0
        assert result_window.params["window"] == 252

    def test_fat_tail_larger_var(self, normal_returns, fat_tail_returns):
        """Fat-tail returns phải có VaR 99% âm hơn Normal."""
        var_normal   = hs_var(normal_returns, 0.99).var
        var_fat_tail = hs_var(fat_tail_returns, 0.99).var
        assert var_fat_tail <= var_normal, \
            "Fat-tail VaR should be more negative than Normal VaR"

    def test_short_series(self):
        """Hoạt động với chuỗi ngắn."""
        r = make_normal_returns(n=50, seed=SEED)
        result = hs_var(r, 0.95)
        assert result.var < 0

    def test_nan_handling(self):
        """Bỏ qua NaN một cách sạch."""
        r = make_normal_returns(n=200, seed=SEED).astype(float)
        r[::10] = np.nan
        result = hs_var(r, 0.95)
        assert result.var < 0
        assert np.isfinite(result.var)


# ---------------------------------------------------------------------------
# Rolling HS VaR
# ---------------------------------------------------------------------------

class TestRollingHSVaR:

    def test_returns_series(self, return_series):
        rv = rolling_hs_var(return_series, confidence=0.95, window=252)
        assert len(rv) == len(return_series)
        # Phần đầu (< window) là NaN
        assert rv.iloc[:251].isna().all()
        # Phần sau có giá trị âm
        assert (rv.dropna() < 0).all()

    def test_monotonicity_across_confidence(self, return_series):
        rv95 = rolling_hs_var(return_series, 0.95, 252).dropna()
        rv99 = rolling_hs_var(return_series, 0.99, 252).dropna()
        assert (rv99 <= rv95).all(), "VaR 99% ≤ VaR 95% at every point"


# ---------------------------------------------------------------------------
# Parametric VaR
# ---------------------------------------------------------------------------

class TestParametricVaR:

    @pytest.mark.parametrize("dist", ["normal", "student_t", "nig"])
    def test_basic(self, normal_returns, dist):
        result = parametric_var(normal_returns, confidence=0.95, dist=dist)
        assert isinstance(result, VaRResult)
        assert result.var < 0
        assert dist in result.method

    def test_normal_vs_empirical(self, normal_returns):
        """Parametric Normal VaR phải gần với HS VaR cho Normal data."""
        var_param = parametric_var(normal_returns, 0.95, "normal").var
        var_hs    = hs_var(normal_returns, 0.95).var
        # Sai lệch dưới 10% relative
        assert abs(var_param - var_hs) / abs(var_hs) < 0.10

    def test_student_t_fatter_tail(self, fat_tail_returns):
        """Student-t fit phải cho VaR 99% âm hơn Normal fit."""
        var_normal = parametric_var(fat_tail_returns, 0.99, "normal").var
        var_t      = parametric_var(fat_tail_returns, 0.99, "student_t").var
        assert var_t <= var_normal, \
            "Student-t VaR should be ≤ Normal VaR for fat-tail data"

    def test_unknown_dist_raises(self, normal_returns):
        with pytest.raises(ValueError, match="Unknown distribution"):
            parametric_var(normal_returns, 0.95, "invalid_dist")

    def test_params_stored(self, normal_returns):
        result = parametric_var(normal_returns, 0.95, "normal")
        assert "mu" in result.params
        assert "sigma" in result.params
        assert result.params["sigma"] > 0

    def test_student_t_df_positive(self, normal_returns):
        result = parametric_var(normal_returns, 0.95, "student_t")
        assert result.params["df"] > 0


# ---------------------------------------------------------------------------
# Cornish-Fisher VaR
# ---------------------------------------------------------------------------

class TestCornishFisherVaR:

    def test_basic(self, normal_returns):
        result = cornish_fisher_var(normal_returns, 0.95)
        assert isinstance(result, VaRResult)
        assert result.var < 0
        assert result.method == "cornish_fisher"

    def test_skewness_adjustment(self):
        """Với return lệch trái mạnh, CF VaR phải âm hơn Normal VaR."""
        rng = np.random.default_rng(SEED)
        # Tạo data lệch trái bằng cách lấy -chi2
        r = -rng.chisquare(df=2, size=N_OBS)
        r = (r - r.mean()) / r.std() * 0.01  # standardise

        var_cf     = cornish_fisher_var(r, 0.99).var
        var_normal = parametric_var(r, 0.99, "normal").var
        # CF nên capture được tail nặng hơn
        assert var_cf <= var_normal * 1.1  # cho phép sai lệch nhỏ

    def test_params_stored(self, normal_returns):
        result = cornish_fisher_var(normal_returns, 0.95)
        assert "skewness" in result.params
        assert "excess_kurtosis" in result.params
        assert "z_cf" in result.params

    def test_near_normal_close_to_parametric(self, normal_returns):
        """Với Normal data, CF phải gần với Parametric Normal."""
        var_cf = cornish_fisher_var(normal_returns, 0.95).var
        var_nm = parametric_var(normal_returns, 0.95, "normal").var
        assert abs(var_cf - var_nm) / abs(var_nm) < 0.05


# ---------------------------------------------------------------------------
# GARCH VaR
# ---------------------------------------------------------------------------

class TestGARCHVaR:

    @pytest.mark.parametrize("dist", ["normal", "student_t"])
    def test_basic(self, garch_returns, dist):
        result = garch_var(garch_returns, 0.95, dist=dist)
        assert isinstance(result, VaRResult)
        assert result.var < 0
        assert dist in result.method

    def test_sigma2_next_positive(self, garch_returns):
        result = garch_var(garch_returns, 0.95)
        assert result.params["sigma2_next"] > 0

    def test_horizon_scaling(self, garch_returns):
        """VaR 10-day phải lebih negatif dari VaR 1-day."""
        var1  = garch_var(garch_returns, 0.95, horizon=1).var
        var10 = garch_var(garch_returns, 0.95, horizon=10).var
        assert var10 < var1, "Longer horizon VaR should be more negative"
        # Heuristic: roughly sqrt(10) scaling
        ratio = var10 / var1
        assert 2.0 < abs(ratio) < 5.0, f"Unexpected scaling ratio: {ratio:.2f}"

    def test_invalid_dist_raises(self, garch_returns):
        with pytest.raises(ValueError):
            garch_var(garch_returns, 0.95, dist="invalid")

    def test_garch_params_valid(self, garch_returns):
        """GARCH params phải thoả mãn stationarity: α+β < 1."""
        result = garch_var(garch_returns, 0.95)
        alpha = result.params["alpha"]
        beta  = result.params["beta"]
        assert alpha + beta < 1.0, "GARCH should be stationary"


# ---------------------------------------------------------------------------
# Filtered Historical Simulation (FHS) VaR
# ---------------------------------------------------------------------------

class TestFilteredHSVaR:

    def test_basic(self, garch_returns):
        result = filtered_hs_var(garch_returns, 0.95)
        assert isinstance(result, VaRResult)
        assert result.var < 0
        assert result.method == "filtered_hs"

    def test_sigma_next_positive(self, garch_returns):
        result = filtered_hs_var(garch_returns, 0.95)
        assert result.params["sigma_next"] > 0

    def test_fhs_vs_hs(self, garch_returns):
        """FHS nên capture volatility clustering tốt hơn plain HS."""
        # Cả hai phải trả về giá trị hợp lệ
        var_fhs = filtered_hs_var(garch_returns, 0.95).var
        var_hs  = hs_var(garch_returns, 0.95).var
        assert var_fhs < 0
        assert var_hs  < 0

    def test_confidence_monotonicity(self, garch_returns):
        var95 = filtered_hs_var(garch_returns, 0.95).var
        var99 = filtered_hs_var(garch_returns, 0.99).var
        assert var99 <= var95


# ---------------------------------------------------------------------------
# compare_var_methods()
# ---------------------------------------------------------------------------

class TestCompareVaRMethods:

    def test_returns_dataframe(self, normal_returns):
        df = compare_var_methods(normal_returns, confidence=0.95)
        assert hasattr(df, "index")
        assert "var" in df.columns

    def test_all_methods_present(self, normal_returns):
        df = compare_var_methods(normal_returns, 0.95)
        expected = {"hs", "parametric_normal", "parametric_t",
                    "cornish_fisher", "garch_normal", "garch_t", "filtered_hs"}
        assert expected.issubset(set(df.index))

    def test_all_negative(self, normal_returns):
        df = compare_var_methods(normal_returns, 0.95)
        finite_vars = df["var"].dropna()
        assert (finite_vars < 0).all(), "All VaR values should be negative"

    def test_sorted_ascending(self, normal_returns):
        """Kết quả nên được sắp xếp từ âm nhất đến ít âm nhất."""
        df = compare_var_methods(normal_returns, 0.95).dropna()
        assert df["var"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Cross-method ordering properties
# ---------------------------------------------------------------------------

class TestVaROrderingProperties:

    def test_99_more_negative_than_95(self, normal_returns):
        """VaR 99% phải âm hơn VaR 95% cho mọi method."""
        for method, fn in [
            ("hs",       lambda r, cl: hs_var(r, cl).var),
            ("normal",   lambda r, cl: parametric_var(r, cl, "normal").var),
            ("t",        lambda r, cl: parametric_var(r, cl, "student_t").var),
            ("cf",       lambda r, cl: cornish_fisher_var(r, cl).var),
            ("garch",    lambda r, cl: garch_var(r, cl).var),
        ]:
            v95 = fn(normal_returns, 0.95)
            v99 = fn(normal_returns, 0.99)
            assert v99 <= v95, f"{method}: VaR99 should ≤ VaR95"

    def test_fat_tail_more_negative_than_normal_at_99(self):
        """Fat-tail returns → VaR 99% phải âm hơn Normal returns VaR 99%."""
        r_normal   = make_normal_returns(n=N_OBS, sigma=0.01, seed=SEED)
        r_fat_tail = make_fat_tail_returns(n=N_OBS, df=3, sigma=0.01, seed=SEED)

        var_n  = hs_var(r_normal,   0.99).var
        var_ft = hs_var(r_fat_tail, 0.99).var
        assert var_ft <= var_n

    def test_larger_sigma_more_negative_var(self):
        """Volatility cao hơn → VaR âm hơn."""
        r_low  = make_normal_returns(n=N_OBS, sigma=0.005, seed=SEED)
        r_high = make_normal_returns(n=N_OBS, sigma=0.020, seed=SEED)

        var_low  = hs_var(r_low,  0.95).var
        var_high = hs_var(r_high, 0.95).var
        assert var_high <= var_low

    def test_var_less_negative_than_min(self, normal_returns):
        """VaR không thể âm hơn giá trị nhỏ nhất."""
        result = hs_var(normal_returns, 0.99)
        assert result.var >= normal_returns.min() - TOL

    def test_var_more_negative_than_mean(self, normal_returns):
        """VaR (1-tail) phải âm hơn mean (với confidence > 50%)."""
        result = hs_var(normal_returns, 0.95)
        assert result.var <= normal_returns.mean() + TOL


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestVaREdgeCases:

    def test_all_positive_returns(self):
        """VaR với returns toàn dương vẫn phải trả về giá trị hợp lệ."""
        r = np.abs(make_normal_returns(n=500, seed=SEED))
        result = hs_var(r, 0.95)
        assert np.isfinite(result.var)

    def test_all_negative_returns(self):
        """VaR với returns toàn âm."""
        r = -np.abs(make_normal_returns(n=500, seed=SEED))
        result = hs_var(r, 0.95)
        assert result.var < 0

    def test_constant_returns(self):
        """Returns constant → VaR phải bằng chính giá trị đó."""
        r = np.full(500, -0.01)
        result = hs_var(r, 0.95)
        assert abs(result.var - (-0.01)) < TOL

    def test_large_n(self):
        """Test với N lớn (100k) không crash."""
        r = make_normal_returns(n=100_000, seed=SEED)
        result = hs_var(r, 0.99)
        assert result.var < 0
        assert np.isfinite(result.var)

    def test_very_high_confidence(self, normal_returns):
        """VaR 99.9% phải hợp lệ."""
        result = hs_var(normal_returns, 0.999)
        assert result.var < 0
        assert np.isfinite(result.var)

    def test_reproducibility(self):
        """Cùng seed → cùng kết quả."""
        r1 = make_normal_returns(n=500, seed=42)
        r2 = make_normal_returns(n=500, seed=42)
        assert np.array_equal(r1, r2)
        assert hs_var(r1, 0.95).var == hs_var(r2, 0.95).var