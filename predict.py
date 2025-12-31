import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from feature_engineering import add_features
from data_fetcher import fetch_historical_streamflow_data, fetch_realtime_streamflow_data
from model import FlashFloodClassifier

def predict_flash_flood(model, scaler, site_number, prediction_date=None, lookback_days=7):
    # If prediction_date is today or None, try real-time data first
    is_today = False
    if prediction_date is None:
        is_today = True
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(prediction_date, "%Y-%m-%d")
        if end_date.date() == datetime.now().date():
            is_today = True

    df = pd.DataFrame()
    if is_today:
        try:
            df = fetch_realtime_streamflow_data(site_number, lookback_days=lookback_days)
            if not df.empty:
                print(f"Using real-time data (iv) for site {site_number}")
        except Exception as e:
            print(f"Error fetching real-time data: {e}. Falling back to historical...")

    if df.empty:
        start_date = end_date - timedelta(days=lookback_days)
        df = fetch_historical_streamflow_data(site_number, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if not df.empty:
            print(f"Using historical data (dv) for site {site_number}")

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
    
    # Final check for NaN or Inf that might have survived feature engineering
    if latest.isnull().any() or np.isinf(latest.values).any():
        latest = latest.replace([np.inf, -np.inf], np.nan).fillna(0)

    try:
        X_scaled = scaler.transform([latest.values])
    except ValueError:
        return None

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        prob = model(X_tensor).item()

    return prob