import pandas as pd
import numpy as np

def normalize_series(series, window=60):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def compute_liquidity_index(metrics_dict):
    df_list = []
    for ticker, metrics in metrics_dict.items():
        normalized_metrics = [normalize_series(m) for m in metrics]
        combined = pd.concat(normalized_metrics, axis=1).mean(axis=1)
        df_list.append(combined)
    liquidity_index = pd.concat(df_list, axis=1).mean(axis=1)
    return liquidity_index.dropna()

def classify_regime(index, normal=0.5, watch=1.5):

    regime = pd.Series("normal", index=index.index, dtype=object)
    regime[index >= normal] = "watch"
    regime[index >= watch] = "stress"
    return regime