# Fat-Tail Modeling: Extreme Value Analysis & Heavy-Tailed Distributions

## Overview
This project studies **fat-tailed behavior in data** and builds a **statistical modeling pipeline** to understand and model extreme events. It focuses on how classical assumptions (e.g., Gaussian distributions) fail in the tails and how alternative approaches provide better fit and reliability.

The work is positioned as **applied statistical data science**, not domain-specific finance.

---

## Problem
Many real-world datasets exhibit **heavy tails**, meaning extreme values occur more frequently than predicted by normal distributions. This leads to:

- Systematic underestimation of extreme events  
- Poor model performance in tail regions  
- Invalid assumptions in standard statistical models  

Key questions:
- How can we detect fat-tailed behavior?
- Which distributions model it better?
- How can we model extremes explicitly?
- How do we validate models in the tail?

---

## Approach

The project follows a structured modeling pipeline:

1. **Data Generation**  
   - Create synthetic datasets (Gaussian vs heavy-tailed)  
   - Control mean/variance to isolate tail effects  

2. **Exploratory Analysis**  
   - Histogram (log scale)  
   - QQ plots  
   - Skewness, kurtosis  

3. **Distribution Modeling**  
   - Fit Normal, Student-t, etc.  
   - Evaluate using likelihood and goodness-of-fit  

4. **Tail Analysis**  
   - Tail index estimation  
   - Extreme quantile behavior  

5. **Extreme Value Theory (EVT)**  
   - Peaks-over-threshold (POT)  
   - Generalized Pareto Distribution (GPD)  

6. **Dependence Modeling**  
   - Copula methods  
   - Tail dependence  

7. **Validation**  
   - Goodness-of-fit tests  
   - Tail-focused QQ plots  
   - Coverage analysis  

8. **Robustness**  
   - Sensitivity to parameters  
   - Stability under extreme scenarios  

---

## Project Structure


.
├── notebooks/ # Analytical workflow (storytelling)
├── src/ # Core statistical modules
├── configs/ # Experiment configurations
├── scripts/ # Reproducible execution
└── README.md


### src/

data/ # Data generation
distributions/ # Distribution models & fitting
tails/ # Tail analysis tools
extreme_value/ # EVT (GPD, POT)
dependence/ # Copula & tail dependence
validation/ # Statistical tests
simulation/ # Monte Carlo processes
evaluation/ # Metrics & uncertainty
pipelines/ # End-to-end workflows
utils/ # Helpers


---

## Methodology

- **Controlled experiments**: isolate tail effects via synthetic data  
- **Model comparison**: evaluate multiple distributions  
- **Tail-focused modeling**: prioritize extreme behavior over global fit  
- **Validation-first mindset**: ensure models perform in the tail  

---

## Reproducibility

Experiments are configuration-driven:

```bash
python scripts/run_modeling.py --config configs/distribution.yaml
```
This ensures:

- Consistent results 
- Easy parameter tuning 
- Separation of logic and experiments

## Key Insights

- Gaussian models underestimate extreme events 
- Heavy-tailed distributions better capture real-world behavior
- EVT provides a principled way to model extremes 
- Tail validation is critical and often overlooked

## Positioning

This project demonstrates:

> Applied statistical modeling for extreme events under heavy-tailed distributions

It emphasizes:

- Strong statistical reasoning 
- Rigorous validation 
- Clean system design