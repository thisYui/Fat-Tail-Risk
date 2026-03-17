"""
src/utils/config_loader.py
---------------------------
Load, merge và validate YAML config files.

Features
--------
  - load_config()      : load YAML → dict, hỗ trợ _base inheritance
  - merge_configs()    : deep-merge nhiều config dict
  - get()              : dotted-key access  cfg.get("monte_carlo.n_paths")
  - ConfigNamespace    : truy cập config như attribute (cfg.monte_carlo.n_paths)
  - validate_config()  : kiểm tra required keys

Usage
-----
>>> from src.utils.config_loader import load_config, ConfigNamespace
>>>
>>> # Load single file
>>> cfg = load_config("configs/simulation.yaml")
>>> print(cfg["monte_carlo"]["n_paths"])           # dict style
>>>
>>> # Load with attribute access
>>> ns = ConfigNamespace(cfg)
>>> print(ns.monte_carlo.n_paths)                  # attribute style
>>>
>>> # Load + merge override
>>> cfg = load_config("configs/risk.yaml", overrides={"backtest.confidence": 0.95})
>>>
>>> # Load multiple configs (child overrides parent)
>>> cfg = load_config("configs/simulation.yaml")   # auto-inherits base.yaml
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_yaml(path: Union[str, Path]) -> dict:
    """
    Load một file YAML thành dict.

    Parameters
    ----------
    path : đường dẫn file YAML

    Returns
    -------
    dict
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep-merge hai config dict.

    override ghi đè base; nested dicts được merge đệ quy.

    Parameters
    ----------
    base     : config gốc
    override : config ghi đè

    Returns
    -------
    dict mới (không thay đổi input)
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = merge_configs(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_config(
    path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Load config YAML, tự động kế thừa từ _base nếu có.

    Parameters
    ----------
    path     : đường dẫn file config chính
    base_dir : thư mục gốc để resolve đường dẫn tương đối (default = cwd)
    overrides: dict dotted-key → value để override sau khi load
               vd {"monte_carlo.n_paths": 100_000, "output.dir": "results"}

    Returns
    -------
    dict config đã merge

    Example
    -------
    >>> cfg = load_config("configs/simulation.yaml",
    ...                   overrides={"monte_carlo.n_paths": 100_000})
    """
    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = Path(base_dir)
    path = Path(path)
    if not path.is_absolute():
        path = base_dir / path

    data = load_yaml(path)

    # Handle _base inheritance
    if "_base" in data:
        base_path_str = data.pop("_base")
        base_path = Path(base_path_str)
        if not base_path.is_absolute():
            base_path = base_dir / base_path
        base_data = load_config(base_path, base_dir=base_dir)
        data = merge_configs(base_data, data)

    # Apply overrides (dotted key)
    if overrides:
        for dotted_key, value in overrides.items():
            _set_dotted(data, dotted_key, value)

    return data


def load_all_configs(
    config_dir: Union[str, Path] = "configs",
    base_file: str = "base.yaml",
) -> Dict[str, dict]:
    """
    Load tất cả các YAML trong config_dir.

    Returns
    -------
    dict {filename_stem: config_dict}
    vd {"base": {...}, "simulation": {...}, "evt": {...}}
    """
    config_dir = Path(config_dir)
    configs = {}
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        try:
            cfg = load_config(yaml_file)
            configs[yaml_file.stem] = cfg
        except Exception as e:
            import warnings
            warnings.warn(f"Could not load {yaml_file}: {e}")
    return configs


# ---------------------------------------------------------------------------
# Dotted key helpers
# ---------------------------------------------------------------------------

def _get_dotted(data: dict, dotted_key: str, default: Any = None) -> Any:
    """
    Truy cập giá trị từ dict bằng dotted key.

    vd _get_dotted(cfg, "monte_carlo.n_paths") → 50000
    """
    keys = dotted_key.split(".")
    cur = data
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _set_dotted(data: dict, dotted_key: str, value: Any) -> None:
    """
    Set giá trị trong dict bằng dotted key (tạo nested nếu cần).

    vd _set_dotted(cfg, "monte_carlo.n_paths", 100_000)
    """
    keys = dotted_key.split(".")
    cur = data
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


# ---------------------------------------------------------------------------
# ConfigNamespace – attribute-style access
# ---------------------------------------------------------------------------

class ConfigNamespace:
    """
    Bọc config dict để truy cập như attribute.

    Example
    -------
    >>> cfg = load_config("configs/simulation.yaml")
    >>> ns = ConfigNamespace(cfg)
    >>> ns.monte_carlo.n_paths          # 50000
    >>> ns.scenarios.active             # ["base", "gfc_2008", ...]
    >>> ns.get("monte_carlo.n_paths")   # 50000 (dotted access)
    >>> ns.get("missing.key", 42)       # 42 (default)
    """

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)
        # Recursively wrap nested dicts
        for key, val in data.items():
            if isinstance(val, dict):
                object.__setattr__(self, key, ConfigNamespace(val))
            else:
                object.__setattr__(self, key, val)

    def __repr__(self) -> str:
        keys = list(self._data.keys())
        return f"ConfigNamespace({keys})"

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Truy cập dotted key với default fallback."""
        return _get_dotted(self._data, dotted_key, default=default)

    def to_dict(self) -> dict:
        """Chuyển ngược về dict gốc."""
        return copy.deepcopy(self._data)

    def override(self, **kwargs) -> "ConfigNamespace":
        """
        Tạo bản copy với các giá trị override.

        Parameters
        ----------
        **kwargs : dotted_key=value pairs

        Example
        -------
        >>> ns2 = ns.override(**{"monte_carlo.n_paths": 100_000})
        """
        new_data = copy.deepcopy(self._data)
        for dotted_key, value in kwargs.items():
            _set_dotted(new_data, dotted_key, value)
        return ConfigNamespace(new_data)


def namespace(path: Union[str, Path], **overrides) -> ConfigNamespace:
    """
    Shortcut: load_config + ConfigNamespace trong một bước.

    Parameters
    ----------
    path     : đường dẫn YAML
    **overrides : dotted_key=value để override

    Example
    -------
    >>> sim = namespace("configs/simulation.yaml",
    ...                 **{"monte_carlo.n_paths": 100_000})
    >>> print(sim.monte_carlo.n_paths)   # 100_000
    """
    cfg = load_config(path, overrides=overrides if overrides else None)
    return ConfigNamespace(cfg)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_config(cfg: dict, required_keys: List[str]) -> None:
    """
    Kiểm tra config có đủ các required_keys không.

    Parameters
    ----------
    cfg           : config dict
    required_keys : danh sách dotted keys bắt buộc

    Raises
    ------
    KeyError nếu thiếu key bắt buộc
    """
    missing = []
    for key in required_keys:
        val = _get_dotted(cfg, key, default=_MISSING)
        if val is _MISSING:
            missing.append(key)
    if missing:
        raise KeyError(
            f"Missing required config keys: {missing}\n"
            "Check your YAML configuration files."
        )


class _MissingType:
    """Sentinel cho missing keys."""
    def __repr__(self):
        return "<MISSING>"


_MISSING = _MissingType()


# ---------------------------------------------------------------------------
# Pipeline config builders
# ---------------------------------------------------------------------------

def build_simulation_config(cfg: dict):
    """Chuyển dict config → SimulationConfig dataclass."""
    from src.pipelines.simulation_pipeline import SimulationConfig
    mc = cfg.get("monte_carlo", {})
    scenarios = cfg.get("scenarios", {})
    portfolio = cfg.get("portfolio", {})
    output = cfg.get("output", {})

    return SimulationConfig(
        scenarios=scenarios.get("active", ["base", "gfc_2008", "covid_2020"]),
        n_paths=mc.get("n_paths", 10_000),
        n_steps=mc.get("n_steps", 252),
        S0=mc.get("S0", 100.0),
        dt=mc.get("dt", 1 / 252),
        seed=mc.get("seed", 42),
        confidence_levels=tuple(mc.get("confidence_levels", [0.90, 0.95, 0.99])),
        run_portfolio=portfolio.get("enabled", False),
        portfolio_weights=portfolio.get("weights"),
        portfolio_mu=portfolio.get("mu_vec"),
        portfolio_cov=portfolio.get("cov_matrix"),
        portfolio_dist=portfolio.get("dist", "normal"),
        output_dir=output.get("dir", "outputs/simulation"),
    )


def build_modeling_config(cfg: dict):
    """Chuyển dict config → ModelingConfig dataclass."""
    from src.pipelines.modeling_pipeline import ModelingConfig
    pot = cfg.get("pot", {})
    rl = cfg.get("return_levels", {})
    evt_risk = cfg.get("evt_risk", {})
    output = cfg.get("output", {})

    return ModelingConfig(
        fit_evt=True,
        threshold_quantile=pot.get("threshold", {}).get("percentile", 0.90),
        confidence_levels=tuple(evt_risk.get("confidence_levels", [0.90, 0.95, 0.99])),
        rank_by="aic",
        output_dir=output.get("dir", "outputs/evt"),
    )


def build_risk_config(cfg: dict):
    """Chuyển dict config → RiskConfig dataclass."""
    from src.pipelines.risk_pipeline import RiskConfig
    portfolio = cfg.get("portfolio", {})
    metrics = cfg.get("metrics", {})
    backtest = cfg.get("backtest", {})
    stress = cfg.get("stress", {})
    output = cfg.get("output", {})

    return RiskConfig(
        risk_free=metrics.get("risk_free_annual", 0.05),
        freq=metrics.get("freq", 252),
        confidence_levels=tuple(
            cfg.get("var", {}).get("confidence_levels", [0.90, 0.95, 0.99])
        ),
        compute_evt=True,
        run_stress=stress.get("parametric", {}).get("enabled", True),
        portfolio_value=portfolio.get("value", 1_000_000),
        mc_stress_n_simulations=stress.get("monte_carlo", {}).get("n_simulations", 10_000),
        mc_stress_horizon=stress.get("monte_carlo", {}).get("horizon_days", 21),
        run_backtest=True,
        backtest_confidence=backtest.get("confidence", 0.99),
        rolling_window=backtest.get("rolling_window", 252),
        output_dir=output.get("dir", "outputs/risk"),
    )