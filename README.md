# Inventory Demand Forecaster

A Streamlit research dashboard for M5 item-store demand. It forecasts 7, 14, or 28 days and estimates reorder needs. It supports the original single-series models and the newer shared global M5 models.

## 1. Model files and upload data

To use the latest global models, copy these four outputs from Kaggle into `models/`:

```text
global_lstm_m5.keras
global_gru_m5.keras
global_model_info.json
global_model_metrics.csv
```

The app switches to global mode when `global_model_info.json` is present. It supports a single M5 item-store series per uploaded CSV, provided that series is listed in the global metadata. Remove that file only if you intend to switch back to the older single-series models.

Create a compatible history CSV for any product-store series in the M5 dataset with this Kaggle cell:

```python
series_id = "HOBBIES_1_001_CA_1_validation"  # change to the product-store series you want
path = "/kaggle/input/m5-forecasting-accuracy"
sales = pd.read_csv(f"{path}/sales_train_validation.csv")
calendar = pd.read_csv(f"{path}/calendar.csv")
prices = pd.read_csv(f"{path}/sell_prices.csv")
row = sales.loc[sales["id"] == series_id].iloc[0]
days = [col for col in sales.columns if col.startswith("d_")]
upload = pd.DataFrame({"d": days, "sales": row[days].to_numpy(dtype=float)})
upload = upload.merge(calendar[["d", "date", "wm_yr_wk", "event_name_1", f"snap_{row['state_id']}"]], on="d", how="left")
item_prices = prices.loc[(prices["item_id"] == row["item_id"]) & (prices["store_id"] == row["store_id"]), ["wm_yr_wk", "sell_price"]]
upload = upload.merge(item_prices, on="wm_yr_wk", how="left")
upload["sell_price"] = upload["sell_price"].ffill().bfill()
upload["is_event"] = upload["event_name_1"].notna().astype(int)
upload["snap"] = upload[f"snap_{row['state_id']}"]
upload["series_id"] = series_id
upload = upload[["series_id", "date", "sales", "sell_price", "is_event", "snap"]]
upload.to_csv("/kaggle/working/website_upload.csv", index=False)
```

Download `website_upload.csv` from Kaggle Output. It should have at least 84 consecutive days. If your `data` DataFrame came from an older preparation cell with missing dates, rerun the corrected module below to generate a clean file.

### Train the global models

Upload the updated `m5_inventory_forecaster.py`, attach the M5 dataset, enable GPU, and run:

```python
import sys
sys.path.insert(0, "/kaggle/input/YOUR-MODULE-DATASET")
from m5_inventory_forecaster import run_global_pipeline

results = run_global_pipeline(
    data_path="/kaggle/input/m5-forecasting-accuracy",
    epochs=12,
    output_dir="/kaggle/working",
)
display(results["metrics"])
```

Download the global model files from Kaggle Output and put them in this project's `models/` folder:

```text
global_gru_m5.keras
global_lstm_m5.keras
global_model_info.json
global_model_metrics.csv
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

In global mode, `series_id` must occur in `global_model_info.json`; each CSV contains one series. `sell_price` should use the same currency and units as training. `is_event` and `snap` are binary flags. Global models directly forecast 28 days, with the app showing the selected first 7, 14, or 28 days. The optional future calendar file is not needed for global inference.

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
