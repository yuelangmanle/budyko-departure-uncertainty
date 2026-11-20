# Cross-Hemisphere Budyko Partitioning and Conformal Uncertainty Quantification

Code and summary outputs accompanying the manuscript:

> **Catchment Structure, Inter-Annual Runoff Persistence, and Physics-Constrained Conformal Uncertainty Across 1,202 Great Britain and Australian Catchments**  
> Zijie Deng, Yu Xia, Jiyang Yu, and Zhenyu Liao

## Overview

This repository contains the reproducible Python workflow across 1,202 rainfall-dominated catchments (52,689 water years) from **CAMELS-GB v1** ($n = 644$) and **CAMELS-AUS v2.03** ($n = 558$):

- Catchment-averaged daily water-year aggregation, unit harmonization, and quality control (including an Australian July–June water-year sensitivity check).
- Orthogonal variance decomposition, Kendall bias-corrected lag-1 autocorrelation, partial rank correlation analysis, closed-form zero-memory transition asymmetry ($\Delta_{\mathrm{null}} = -4\,\mathrm{Cov}(\Delta Q, F_P(P))$), and strictly fold-isolated 5-fold time-block cross-fitted climate counterfactuals with 500-iteration catchment-clustered bootstrap intervals.
- Forcing-conditioned runoff-ratio ($Q/P$) benchmarks across four validation regimes (S1 gauged temporal, S2 5-fold ungauged PUB, S3 cross-continent transfer, and S4 Australia's Millennium Drought) comparing physical Fu curves, gradient-boosted trees, physics-anchored residual learning, and source-calibrated out-of-support damping.
- Conformal uncertainty quantification comparing Split-CP, Local-CP, CQR, Mondrian CQR, Logit-manifold CQR, and sequential Online Adaptive Conformal Inference (ACI).

## Repository Structure

```text
.
├── scripts/
│   ├── 01_build_dataset.py               # Water-year aggregation and quality control (GB + AUS)
│   ├── 02_diagnostics_and_memory.py      # Variance decomposition, Theorem 1 null verification, and BFI partial correlations
│   ├── 03_models_and_conformal_uq.py     # S1–S4 baselines and Split-CP / Local-CP / CQR / Mondrian CQR
│   ├── 03b_hybrid_and_adaptive_uq.py     # Physics-anchored residual HGBR, Logit-CQR, Online ACI, and 5-fold feature importance
│   ├── 04_crossfit_null_and_transfer.py  # Fold-isolated 5-fold cross-fitted climate nulls, basin bootstrap, and S3 M0/damped baselines
│   └── 05_plot_figures.py                # Generates all six 600 DPI figures (Fig1.png–Fig6.png)
├── data/
│   └── gb_aus_basin_attributes.csv       # Harmonized static attributes and long-term indices for 1,202 catchments
├── results/
│   ├── dataset_summary.json
│   ├── diagnostics_summary.json
│   ├── modeling_uq_summary.json
│   ├── modeling_extended_summary.json
│   └── crossfit_null_and_transfer.json
├── figures/
│   └── Fig1.png ... Fig6.png             # Six 600 DPI manuscript figures
├── REPRODUCE.md
└── requirements.txt
```

## Input Data

Raw daily hydrometeorological archives are publicly available from their official repositories:

- **CAMELS-GB v1**: Coxon et al. (2020), UK Environmental Information Data Centre, https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9
- **CAMELS-AUS v2.03**: Fowler et al. (2024, 2025), Zenodo, https://doi.org/10.5281/zenodo.14289037

See `REPRODUCE.md` for step-by-step execution instructions.

## License

MIT License (see `LICENSE`).
