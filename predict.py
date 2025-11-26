import torch
import pandas as pd
from datetime import datetime, timedelta

from feature_engineering import add_features
from data_fetcher import fetch_historical_streamflow_data
from model import FlashFloodClassifier

def predict_flash_flood(model, scaler, site_number, prediction_date=None, lookback_days=7):
    if prediction_date is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(prediction_date, "%Y-%m-%d")

    start_date = end_date - timedelta(days=lookback_days)

    df = fetch_historical_streamflow_data(site_number, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if df.empty:
        print("Not enough data for prediction.")
        return None

    df = add_features(df)

    if df.empty:
        print("Not enough data for prediction after processing.")
        return None

    features = ['log_streamflow', 'streamflow_p10', 'streamflow_p50',
                'streamflow_p90', 'streamflow_diff', 'streamflow_pct_change']

    latest = df.set_index("date_time")[features].asof(end_date)
    X_scaled = scaler.transform([latest.values])
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        prob = model(X_tensor).item()

    return prob