# Fat-Tail Risk Modeling & Extreme Value Analysis

## 1. Overview

This project explores **fat-tail risk in financial returns** using advanced statistical modeling techniques.
Instead of assuming normality, it focuses on **heavy-tailed distributions**, **Extreme Value Theory (EVT)**, and **copula-based dependence modeling** to better capture tail risk.

The pipeline covers the full lifecycle:

```
Simulation → Distribution Fitting → Tail Analysis → EVT → Copula → Portfolio Risk → Stress Testing → Backtesting
```

---

## 2. Objectives

* Model **non-Gaussian financial returns**
* Estimate tail risk using EVT (Peaks Over Threshold)
* Compare risk metrics:

  * Value at Risk (VaR)
  * Conditional Value at Risk (CVaR)
* Capture dependency structure using copulas
* Validate models via backtesting
* Analyze robustness under stress scenarios

---

## 3. Project Structure

```
Fat-Tail-Risk/
│
├── data/                     # Simulated & processed datasets
│
├── notebooks/                # Research narrative (main entry point)
│   ├── 01_simulation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_distribution_fitting.ipynb
│   ├── 04_tail_analysis.ipynb
│   ├── 05_evt_model.ipynb
│   ├── 06_copula.ipynb
│   ├── 07_portfolio_risk.ipynb
│   ├── 08_stress_testing.ipynb
│   ├── 09_backtesting.ipynb
│   └── 10_reporting.ipynb
│
├── src/
│   ├── models/               # Statistical models
│   │   ├── var_models.py
│   │   ├── cvar_models.py
│   │   ├── evt.py
│   │   ├── copula.py
│   │   └── distribution.py
│   │
│   ├── simulation/           # Data generation
│   ├── features/             # Feature engineering
│   ├── risk/                 # Risk metrics & backtesting
│   ├── visualization/        # Plots & reporting
│   └── pipelines/            # End-to-end workflows
│
├── configs/                  # Parameter configuration
├── experiments/              # Experiment tracking (metrics + artifacts)
├── scripts/                  # CLI entrypoints
├── tests/                    # Unit tests for core logic
│
└── README.md
```

---

## 4. Key Concepts

### 4.1 Fat-Tailed Distributions

Financial returns often exhibit:

* heavy tails
* skewness
* excess kurtosis

We compare:

* Normal
* Student-t
* Pareto
* (optional) α-stable distributions

---

### 4.2 Extreme Value Theory (EVT)

We use **Peaks Over Threshold (POT)** to model tail behavior:

* Tail modeled with Generalized Pareto Distribution (GPD)
* Threshold selection is critical

---

### 4.3 Risk Metrics

* **VaR (Value at Risk)**: quantile-based risk measure
* **CVaR (Expected Shortfall)**: expected loss beyond VaR

These are implemented independently to highlight their differences.

---

### 4.4 Copula Modeling

Captures **dependency structure** between assets beyond linear correlation:

* Gaussian Copula
* t-Copula

---

### 4.5 Backtesting

Model validation includes:

* Violation rate analysis
* Kupiec test
* Conditional coverage tests

---

## 5. How to Run

### 5.1 Install dependencies

```
pip install -r requirements.txt
```

---

### 5.2 Run notebooks (recommended)

Start with:

```
notebooks/01_simulation.ipynb
```

and follow the pipeline sequentially.

---

### 5.3 Run via CLI (optional)

```
python scripts/run_full_pipeline.py --config configs/base.yaml
```

---

## 6. Experiment Tracking

Each experiment is stored in:

```
experiments/exp_xxx/
```

Includes:

* configuration
* metrics (VaR, CVaR, violation rate)
* generated plots

This ensures reproducibility and avoids selective reporting.

---

## 7. Testing

Basic unit tests are provided for:

* VaR monotonicity
* CVaR ≥ VaR
* EVT consistency

Run:

```
pytest tests/
```

---

## 8. Highlights

* End-to-end **tail risk modeling pipeline**
* Clear separation between:

  * research (notebooks)
  * logic (src)
  * experiments
* Focus on **statistical rigor**, not just ML
* Designed for **reproducible research**

---

## 9. Future Improvements

* Multivariate EVT extensions
* Dynamic copula models
* Real market data integration
* Bayesian tail estimation

---

## 10. Author
