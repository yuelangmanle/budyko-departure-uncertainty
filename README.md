# Cross-Hemisphere Budyko Partitioning and Conformal Uncertainty Quantification

Code and summary outputs accompanying the manuscript:

> **Separating Catchment Structure, Inter-Annual Storage Memory, and Heteroscedastic Conformal Uncertainty in Annual Budyko Partitioning Across 1,202 Great Britain and Australian Catchments**  
> Zijie Deng, Yu Xia, Jiyang Yu, and Zhenyu Liao

## Overview

This repository implements the data preprocessing, variance decomposition, permutation-null wetting--drying asymmetry tests, zero-leakage forecasting experiments, and conformal prediction benchmarks across 1,202 low-snow catchments (52,689 water years) from **CAMELS-GB v1** ($n = 644$) and **CAMELS-AUS v2.03** ($n = 558$).

## Repository Structure

```text
.
├── scripts/
│   ├── 01_build_dataset.py              # Water-year aggregation and quality control (GB + AUS)
│   ├── 02_diagnostics_and_asymmetry.py  # Variance partitioning, lag-1 memory, and permutation-null tests
│   ├── 03_models_and_conformal_uq.py    # Zero-leakage S1-S4 benchmarks and Split-CP / Local-CP / CQR / Mondrian CQR
│   └── 04_generate_figures.py           # Generates publication figures (Figs. 1-5)
├── data/
│   └── gb_aus_basin_attributes.csv      # Harmonized static attributes and long-term indices for 1,202 catchments
├── results/
│   ├── dataset_summary.json             # Quality-control filter counts and regional statistics
│   ├── diagnostics_summary.json         # Variance decomposition and asymmetry test outputs
│   └── modeling_uq_summary.json         # Out-of-sample RMSE, R2, coverage, width, and Winkler scores
├── figures/
│   └── fig1_*.png ... fig5_*.png        # 300 DPI manuscript figures
└── requirements.txt
```

## Input Data

Raw daily hydrometeorological archives are not redistributed here and can be obtained directly from their official repositories:

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
python scripts/04_generate_figures.py
```

## License

MIT License (see `LICENSE`).
