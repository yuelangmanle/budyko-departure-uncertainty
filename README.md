# Cross-Hemisphere Budyko Partitioning and Conformal Uncertainty Quantification

Code and summary outputs accompanying the manuscript:

> **Separating Catchment Structure, Inter-Annual Storage Memory, and Physics-Anchored Conformal Uncertainty in Annual Budyko Partitioning Across 1,202 Great Britain and Australian Catchments**  
> Zijie Deng, Yu Xia, Jiyang Yu, and Zhenyu Liao

## Overview

This repository provides the full reproducible pipeline across 1,202 low-snow catchments (52,689 water years) from **CAMELS-GB v1** ($n = 644$) and **CAMELS-AUS v2.03** ($n = 558$), including:

- Official daily catchment-averaged water-year aggregation and quality control (with Australian July--June water-year sensitivity check).
- Orthogonal variance decomposition, partial rank correlation analysis of catchment memory drivers, and closed-form analytical zero-memory transition asymmetry verification ($\Delta_{\mathrm{null}} = -4\,\mathrm{Cov}(\Delta Q, F_P(P))$).
- Zero-leakage benchmarks across four validation scenarios (gauged temporal, 5-fold ungauged PUB, cross-continent transfer, and Australia's Millennium Drought) comparing physical Fu curves, direct gradient-boosted trees, and physics-anchored residual learning.
- Conformal uncertainty quantification comparing Split-CP, Local-CP, CQR, Mondrian CQR, boundary-preserving Logit-manifold CQR, and sequential Online Adaptive Conformal Inference (ACI).

## Repository Structure

```text
.
├── scripts/
│   ├── 01_build_dataset.py              # Water-year aggregation and quality control (GB + AUS)
│   ├── 02_diagnostics_and_asymmetry.py  # Variance decomposition, Theorem 1 null verification, and BFI partial correlations
│   ├── 03_models_and_conformal_uq.py    # Baseline S1-S4 benchmarks and Split-CP / Local-CP / CQR / Mondrian CQR
│   ├── 03_models_upgraded_suite.py      # Physics-anchored residual HGBR, Logit-CQR, Online ACI, and feature importance
│   └── 04_generate_figures.py           # Generates all six 300 DPI manuscript figures (Figs. 1-6)
├── data/
│   └── gb_aus_basin_attributes.csv      # Harmonized static attributes and long-term indices for 1,202 catchments
├── results/
│   ├── dataset_summary.json
│   ├── diagnostics_summary.json
│   ├── diagnostics_upgraded.json
│   ├── modeling_uq_summary.json
│   └── modeling_upgraded.json
├── figures/
│   └── fig1_*.png ... fig6_*.png        # Six 300 DPI manuscript figures
└── requirements.txt
```

## Input Data

Raw daily hydrometeorological archives can be downloaded from their official repositories:

- **CAMELS-GB v1**: Coxon et al. (2020), UK Environmental Information Data Centre, https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9
- **CAMELS-AUS v2.03**: Fowler et al. (2024), Zenodo, https://doi.org/10.5281/zenodo.14289037

## Reproducing the Analysis

```bash
pip install -r requirements.txt

export CAMELS_GB_ZIP="/path/to/CAMELS_GB_v1.zip"
export CAMELS_AUS_DIR="/path/to/camels_aus_v2.03"

python scripts/01_build_dataset.py
python scripts/02_diagnostics_and_asymmetry.py
python scripts/03_models_and_conformal_uq.py
python scripts/03_models_upgraded_suite.py
python scripts/04_generate_figures.py
```

## License

MIT License (see `LICENSE`).
