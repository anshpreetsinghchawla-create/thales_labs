import yfinance as yf 
import pandas as pd
from datetime import datetime

def fetch_yahoo_data(tickers, start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d')):
    data = {}
    for ticker in tickers:

        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df.dropna()
    return data

