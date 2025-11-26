import pandas as pd

def add_features(df, window_size='7D'):
    df = df.copy()

    # Ensure correct dtypes
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['streamflow_cfs'] = pd.to_numeric(df['streamflow_cfs'], errors='coerce')

    df.set_index('date_time', inplace=True)
    df.sort_index(inplace=True)

    # Rolling percentiles
    df['streamflow_p10'] = df['streamflow_cfs'].rolling(window=window_size).quantile(0.10)
    df['streamflow_p50'] = df['streamflow_cfs'].rolling(window=window_size).quantile(0.50)
    df['streamflow_p90'] = df['streamflow_cfs'].rolling(window=window_size).quantile(0.90)

    # Temporal gradients
    df['streamflow_diff'] = df['streamflow_cfs'].diff()
    df['streamflow_pct_change'] = df['streamflow_cfs'].pct_change()

    # Drop missing values
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    return df