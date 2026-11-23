# Zimbabwe House Price Prediction

This project collects public Zimbabwe house-listing data, cleans and explores the listings, and compares a linear baseline with a multimodal price model. Prices are treated as **USD**, including listings whose currency field is blank, based on the project owner's confirmation. These are advertised asking prices, not verified sale prices.

## Project contents

- `data/houses.csv` — collected source dataset (2,646 listing rows).
- `data_dictionary.csv` — field descriptions and example values for the dataset.
- `images/` — downloaded house photos in WebP format. Filenames use `<listing_id>_<photo_id>.webp`; the numeric filename prefix maps to `listing_id` in `data/houses.csv`. One legacy filename zero-pads listing ID `282` as `0282`. Some listings have no locally available photo.
- `src/scrape_listings.py` — collection code for the source listing sitemap or search pages.
- `src/train_models.py` — training, hold-out evaluation, residual analysis, and report-artifact generation.
- `models/` — trained Ridge coefficients (`linear_baseline.npz`), multimodal PyTorch weights (`multimodal_fusion.pt`), and model metadata. The project does not use Joblib for model serialization.
- `notebooks/01_clean_explore.ipynb` — deduplication, USD-price preparation, missing-value/outlier review, summary tables, and plots.
- `notebooks/02_fit_models.ipynb` — Ridge baseline and multimodal model training and assessment.
- `reports/eda/` — cleaned data, data-quality summaries, and EDA plots.
- `reports/modeling/` — model scores, predictions, residual tables, and diagnostic plots.
- `reports/final_report.md` — concise summary of methods, findings, and limitations.

## Set up the environment

Use Python 3.11 (the notebooks were configured for Python 3.11.9).

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name zimbabwe-houses --display-name "Zimbabwe houses (.venv)"
python src\check_environment.py
```

In VS Code, open the notebooks and select the `Zimbabwe houses (.venv)` kernel. On macOS/Linux, activate the environment with `source .venv/bin/activate` and use forward-slash paths in commands.

## Reproduce the analysis

1. Run `notebooks/01_clean_explore.ipynb` from top to bottom. This reads `data/houses.csv` without overwriting it and creates cleaned data and EDA outputs under `reports/eda/`.
2. Run `notebooks/02_fit_models.ipynb` from top to bottom. This creates the seeded train/validation/test split, trains both models, and writes model results and diagnostics under `reports/modeling/`.

The model run can also be started from a terminal after the cleaned data exists:

```powershell
python src\train_models.py --epochs 60 --seed 42
```

The test set contains 522 listings and is reserved for final evaluation. Do not use it to tune model choices or preprocessing.

The models are trained and saved by this command or by running the modeling notebook. The Ridge model is serialized as NumPy arrays because its fitted coefficients and vocabulary are stored directly; the multimodal network is saved as a PyTorch checkpoint. A `.joblib` file is not expected.

## Collect or refresh listings

Collection requires an internet connection and is **not** needed to reproduce the included analysis. Check the source site's terms and `robots.txt` before collecting; the script enforces the site's robots rules and a minimum one-second request delay. The following example collects up to 25 new listings and one photo per listing:

```powershell
python src\scrape_listings.py --source sitemap --max-listings 25 --images-per-listing 1 --delay 2
```

The script appends newly collected records to `data/houses.csv` and stores photos in `images/`. It skips IDs already present. The included snapshot was collected on 2026-11-22; live listings and resulting metrics may change when data is refreshed. See `python src\scrape_listings.py --help` for all options.

## Results at a glance

On the current held-out split, the multimodal model obtains MAE $104,521, RMSE $217,604, and R² 0.624; the Ridge baseline obtains MAE $109,676, RMSE $234,135, and R² 0.565. The modest gain is based on one split; errors are much larger for the most expensive listings. The evaluation and caveats are documented in [`reports/final_report.md`](reports/final_report.md).
