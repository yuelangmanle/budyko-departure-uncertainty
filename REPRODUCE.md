# Reproducibility Guide (Anonymized Review Bundle)

## Environment
Tested on Python 3.12 with pinned scientific dependencies (see `requirements.txt`):
```bash
pip install -r requirements.txt
```

## Input Data (Official Public Archives)
Download the two official catchment-averaged benchmarks and place them under `data/raw/`:
1. **CAMELS-GB v1**: https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9 (`CAMELS_GB_v1.zip`)
2. **CAMELS-AUS v2.03**: https://doi.org/10.5281/zenodo.14289037 (`camels_aus_v2.03/`)

## Execution Order
Run the pipeline from the bundle root directory:
```bash
python scripts/01_build_dataset.py
python scripts/02_diagnostics_upgraded.py
python scripts/03_models_and_conformal_uq.py
python scripts/03_models_upgraded.py
python scripts/07_p0_rigorous_checks.py
python scripts/05_generate_top_journal_figures.py
```

All pre-computed JSON summary tables corresponding to Tables 1–4 and Figures 1–6 are included in `results/`.
