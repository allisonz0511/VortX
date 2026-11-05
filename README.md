The model VortX forecasts the 15-day mean zonal wind at 60°N, 10 hPa, at leads of 5–30 days, and scores the forecasts against 32 sudden stratospheric warming (SSW) onsets from 1979–2025.

vortx.py contains the full machine learning pipeline, which includes ERA5 preprocessing (100 hPa EP flux, 10 hPa wave-1/2 Fourier coefficients, ū60), Charlton–Polvani SSW detection, feature construction, XGBoost training with leave-one-out-cross-validation, scoring (AUC, hit rate, false-alarm rate vs. persistence), ablation, SHAP, and all figures.

Run order: build_full_dataset() --> build_base() --> table_skill() --> table_onset() --> table_ablation() --> table_composite() --> table_shap() --> make_figures()

Data: ERA5 daily 00 UTC u, v, T, Φ at 10–100 hPa, 20–90°N, 1° grid, one file per month in data/raw/era5_YYYY_MM.nc (download from the Copernicus Climate Data Store). The QBO index is fetched from NOAA PSL.

Requires: numpy, pandas, xarray, xgboost, shap, scikit-learn, matplotlib, cartopy, aostools.
