import pandas as pd
from data_fetcher import fetch_historical_streamflow_data
from feature_engineering import add_features
from train import train_and_evaluate
from predict import predict_flash_flood
from model import FlashFloodClassifier
import torch
import joblib

def main():
    # Example: Fetch data for Brazos River
    site_number = "08166250"
    df = fetch_historical_streamflow_data(site_number, "2020-01-01", "2025-07-02")
    df = add_features(df)

    features = ['streamflow_cfs', 'streamflow_p10', 'streamflow_p50',
                'streamflow_p90', 'streamflow_diff', 'streamflow_pct_change']

    # Train and evaluate
    model, scaler, metrics = train_and_evaluate(df, features)

    # Load model and scaler
    loaded_model = FlashFloodClassifier(input_dim=len(features))
    loaded_model.load_state_dict(torch.load("flash_flood_model.pth"))
    loaded_scaler = joblib.load("scaler.pkl")

    # Prediction example
    probability = predict_flash_flood(loaded_model, loaded_scaler, site_number, prediction_date="2025-07-04")
    print(f"Prediction for site {site_number} on 2025-07-04: {probability:.4f}")

if __name__ == "__main__":
    main()