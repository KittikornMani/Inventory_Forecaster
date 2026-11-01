# Inventory Demand Forecaster

A Streamlit research dashboard for one M5 item-store time series. It loads the LSTM and GRU models, forecasts 7, 14, or 28 days, and estimates when stock reaches a reorder threshold. It supports the supplied 28-day, 7-feature models and the updated 56-day, 14-feature Kaggle training module.

## 1. Model files and upload data

The four files from `Models.zip` are already in `models/`. The supplied models expect a 28-day window and 7 input features. `models/model_info.json` names the series shown in your Kaggle screenshots; check it against the actual series used to train the models.

The app also needs historical daily data. If the Kaggle notebook still has your `data` DataFrame, create the upload CSV in a Kaggle cell:

```python
upload = data[["date", "sales", "sell_price", "is_event", "snap"]].copy()
upload.insert(0, "series_id", "HOBBIES_1_001_CA_1_validation")
upload.to_csv("/kaggle/working/website_upload.csv", index=False)
```

Download `website_upload.csv` from Kaggle Output. It should have at least 84 consecutive days. If your `data` DataFrame came from an older preparation cell with missing dates, rerun the corrected module below to generate a clean file.

### Optional: retrain with corrected daily history

The previous Kaggle module could drop non-event days during training. To make a stronger, consistent new experiment, upload the updated `m5_inventory_forecaster.py`, attach the M5 dataset, and run:

```python
import sys
sys.path.insert(0, "/kaggle/input/YOUR-MODULE-DATASET")
from m5_inventory_forecaster import run_pipeline, show_results

results = run_pipeline(
    data_path="/kaggle/input/m5-forecasting-accuracy",
    series_id="HOBBIES_1_001_CA_1_validation",
    epochs=30,
    output_dir="/kaggle/working",
)
show_results(results)
```

Download the following files from Kaggle Output and replace the files in this project's `models/` folder:

```text
gru_inventory_forecaster.keras
lstm_inventory_forecaster.keras
feature_scaler.pkl
target_scaler.pkl
model_info.json
```

Use the Kaggle output `website_upload.csv` as your first website upload. The small `sample_data/upload_template.csv` illustrates columns only; it has too few rows to run a forecast.

If your Kaggle module was uploaded as a Kaggle dataset, create a **new dataset version** after updating the `.py` file, then restart the notebook session and rerun training. Existing notebook imports may still refer to the old code.

## 2. Run the website locally

Use Python 3.11 or 3.12 in a virtual environment. From this folder:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens in your browser. Choose a model, upload `website_upload.csv`, set current stock and lead time, then click **Generate forecast**.

## 3. Put it online

Push this folder to a private GitHub repository, with the five model files inside `models/`. In Streamlit Community Cloud, create an app from the repository and set the entrypoint to `app.py`. Check that the model files are permitted by your repository and hosting limits. Avoid publishing private business sales data or model artifacts publicly without permission.

This project has not been deployed from this workspace because no Streamlit hosting connection or trained model files were supplied here.

## CSV contract

The upload must have one row per consecutive calendar day and at least 84 rows:

```text
series_id,date,sales,sell_price,is_event,snap
HOBBIES_1_001_CA_1_validation,2016-04-20,1,8.50,0,0
```

`series_id` must match `models/model_info.json`. The trained model only applies to that series. `sell_price` should use the same currency and units as training. `is_event` and `snap` are binary flags. The optional future calendar CSV has `date,is_event,snap` for every forecast date. If omitted, the app assumes no future event/SNAP days and carries forward the last price.

The forecast reports expected demand, which may be fractional. Suggested order quantity rounds up to whole units. The stock chart assumes no deliveries during the horizon. The safety buffer is a user-chosen percentage of lead-time demand, not a statistically calibrated service level. Accuracy results must be judged from held-out test days; the earlier one-series WAPE exceeded 100%, so inventory recommendations are demonstrations rather than validated operational decisions.

## Files

```text
app.py              Streamlit interface
forecasting.py      Upload checks, features, recursive prediction
inventory.py        Inventory calculations
models/             Trained files and series metadata
sample_data/        CSV column example
requirements.txt    Python packages
```
