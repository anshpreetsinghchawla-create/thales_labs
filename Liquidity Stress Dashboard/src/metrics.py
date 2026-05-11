import pandas as pd
import numpy as np

def intraday_range_ratio(df):
    return (df['High'] - df['Low']) / df['Close']

def realized_volatility(df, window=20):
    returns = df['Close'].pct_change()
    return returns.rolling(window).std()

def volume_zscore(df, window=20):
    return (df['Volume'] - df['Volume'].rolling(window).mean()) / df['Volume'].rolling(window).std()


def price_impact(df):
    returns = df['Close'].pct_change().abs()
    return returns /df['Volume']


