# Ranking-system-via-XG-boost
# XGBoost Model Pipeline (scikit-learn)

This project contains a Jupyter notebook that builds an end-to-end ML pipeline using:
- pandas / numpy for data handling
- scikit-learn for preprocessing (imputation, scaling, encoding) via `ColumnTransformer` + `Pipeline`
- XGBoost (`XGBClassifier`) for model training
- generation of a `submission.csv` with predicted probabilities

## Project structure

- `xgb_model_pipeline.ipynb` — main notebook (rename from `xgb_model_pipeline (1).ipynb` for cleanliness)
- `requirements.txt` — Python dependencies
- `submission.csv` — output predictions (generated after running the notebook)

## Data files

The notebook currently loads local parquet files (Windows paths), e.g.:

- `train_data.parquet`
- `add_event.parquet`

To make the notebook portable, place these files in a local `data/` folder and update the paths accordingly.

Suggested layout:
- `data/train_data.parquet`
- `data/add_event.parquet`

## How to run

1. Create and activate an environment (example with conda):
   ```bash
   conda create -n xgb-pipeline python=3.11 -y
   conda activate xgb-pipeline
   pip install -r requirements.txt
   ```

2. Launch Jupyter:
   ```bash
   jupyter lab
   ```

3. Open `xgb_model_pipeline.ipynb` and run all cells.

## Notes / TODO

- The notebook currently sets `test = train_data.parquet` (same as train). If you have a separate test set, load it instead.
- Consider adding:
  - train/valid split (or CV)
  - early stopping with a validation set
  - model + preprocessor persistence (joblib)
  - basic data checks / schema validation

## Output

The notebook writes:
- `submission.csv` with columns: `id1, id2, id3, id5, pred`
