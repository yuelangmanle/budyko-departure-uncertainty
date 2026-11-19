# Antecedent Budyko Departure, Inter-Annual Runoff Asymmetry, and Predictive Uncertainty

Code, processed observational dataset, benchmark tables, and figure scripts accompanying the manuscript:

> **Antecedent Budyko Departure, Inter-Annual Runoff Asymmetry, and Predictive Uncertainty Across 108 Temperate Catchments**  
> Zijie Deng¹, Yu Xia¹*, Jiyang Yu¹, Zhenyu Liao²  
> ¹ Jiangxi Provincial Key Laboratory of Water Resources Allocation and Efficient Utilization, Jiangxi University of Water Resources and Electric Power, Nanchang 330099, China  
> ² School of Geography and Environment, Jiangxi Normal University, Nanchang 330022, China

## Overview

This repository provides a reproducible Python workflow for diagnosing steady-state Budyko transferability breakdown and evaluating physics-guided runoff models and conformal prediction intervals across **108 temperate USGS reference catchments (2005–2024; 1,995 valid station-years; 1,819 contiguous two-year transitions)**, together with two controlled synthetic physical benchmarks.

Key components implemented here:

1. **Antecedent Budyko Departure & Null-Model Separation**: Computes $\Pi_{M,t} = Q_{t-1}/P_{t-1} - B(AI_{t-1})$ and separates persistent cross-decade catchment structure (Fu–Budyko parameter $w_i$) from inter-annual wetting–drying runoff asymmetry using matched memoryless and circular autocorrelation-preserving nulls, with spatial-scale ($A \le 300\;\mathrm{km}^2$) and baseflow ($LFR > 0.25$) stratifications on the 60 direct $0.1^\circ$ ERA5-Land gauges.
2. **Paired Hydrological Transfer Scenarios**:
   - **Scenario 1 (Gauged Regime Extrapolation)**: Humid calibration ($AI \le 0.85$) to drier evaluation ($AI > 0.85$, $n = 1{,}197$ across 92 basins), comparing a two-parameter analytical operator ($M_0^{\mathrm{hyst}}$), 1-year persistence, unconstrained linear ARX baselines (`ARX-Raw`, `ARX-Matched`), and gradient-boosted residual trees.
   - **Scenario 2 (Basin-Isolated PUB)**: 5-fold `GroupKFold` across all 107 contiguous basins ($n = 1{,}197$) and standalone 60-direct-gauge refit ($n = 650$), plus three-way basin isolation via inner `GroupShuffleSplit`.
   - **Scenario 3 (Causal Temporal Transfer)**: 2005–2015 training to 2016–2024 testing ($n = 545$) for both rolling 1-year hindcasts (3A) and multi-year purely meteorological projections (3B).
3. **Shift-Aware Conformal Intervals**: Evaluates exact finite-sample Split Conformal Prediction (`Std-CP`), Jacobian-Only drift adjustment, and tuning-free Integrated Jacobian Sensitivity Conformal Prediction (`IJS-CP`), scored by empirical coverage, mean width, and Winkler Interval Score ($IS_{0.10}$).

## Repository Layout

```text
budyko-departure-uncertainty/
├── data/
│   └── usgs_108_catchments_2005_2024.json   # 20-year quality-controlled USGS + ERA5-Land dataset
├── scripts/
│   ├── fetch_usgs_era5.py                   # USGS NWIS & ERA5-Land retrieval and QC pipeline
│   ├── analyze_catchments.py                # 108-basin analysis, nulls, Scenarios 1–3, cluster bootstraps
│   ├── run_synthetic_benchmarks.py          # Synthetic Benchmarks A (1D ADR) and B (Fu–Budyko runoff)
│   ├── run_sensitivity.py                   # Sensitivity sweeps (k-NN, boundary cutoff, observation noise)
│   ├── plot_main_figures.py                 # Generates Figures 1–4 (300 DPI)
│   └── plot_synthetic_figure.py             # Generates Figure 5 (300 DPI)
├── results/
│   ├── table1_catchment_benchmarks.csv      # Table 1 (22 rows across Scenarios 1–3)
│   ├── table_s1_diagnostics_ablations.csv   # Supplementary Table S1 (16 diagnostic & ablation rows)
│   ├── basin_process_summary.csv            # Basin-level physiographic and coverage summary (92 OOD basins)
│   ├── table2_synthetic_adr.csv             # Table 2 (Benchmark A)
│   ├── table3_synthetic_runoff.csv          # Table 3 (Benchmark B)
│   └── table4_sensitivity.csv               # Table 4 (Sensitivity sweeps)
├── figures/
│   ├── fig1_framework.png
│   ├── fig2_spatial_process_signatures.png
│   ├── fig3_phase_diagram_hydrographs.png
│   ├── fig4_benchmark_interval_quality.png
│   └── fig5_synthetic_benchmarks.png
├── requirements.txt
└── LICENSE
```

## Quick Start

Install dependencies (Python 3.9–3.13):

```bash
pip install -r requirements.txt
```

Run the full analysis and regenerate all tables and 300-DPI figures from the included observational dataset:

```bash
python3 scripts/analyze_catchments.py
python3 scripts/run_synthetic_benchmarks.py
python3 scripts/run_sensitivity.py
python3 scripts/plot_main_figures.py
python3 scripts/plot_synthetic_figure.py
```

### Reproducibility Notes
- Analytical (`M0_hyst`), linear ARX, 1-year persistence, and 1D steady-state baselines reproduce to `0.0000` across Python 3.9–3.13, NumPy 1.x–2.x, and scikit-learn 1.5–1.9.
- Multi-feature `GradientBoostingRegressor` models reproduce within `< 0.0020` RMSE and `< 0.7` percentage points coverage across `scikit-learn` versions.

## Data Sources

- **Streamflow**: U.S. Geological Survey (USGS) National Water Information System (NWIS), daily mean discharge (`00060`) for 108 HCDN / GAGES-II reference gauges (2005–2024).
- **Meteorology**: Copernicus Climate Change Service (C3S) ERA5-Land ($0.1^\circ$ grid), daily precipitation and FAO-56 Penman–Monteith reference evapotranspiration (60 direct-grid gauges and 48 3-donor IDW expansion gauges).

## License

Released under the [MIT License](LICENSE).
