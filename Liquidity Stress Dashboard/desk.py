import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_fetch import fetch_yahoo_data
from src.metrics import intraday_range_ratio, realized_volatility, volume_zscore, price_impact
from src.indices import normalize_series, compute_liquidity_index, classify_regime
from src.lead_lag import lead_lag_correlation

st.set_page_config(layout = "wide")
st.title("FICC Dashboard for Junior Traders")

#Relevant Tickers
curve_tickers = ['^TYX','^TNX','^FVX', '2YY=F', '^IRX']
currency_tickers = ['DX-Y.NYB','EURUSD=X', 'GBPUSD=X', 'AUDUSD=X']
credit_tickers = ['HYG', 'LQD', 'JNK']
rates_tickers = ['SHY', 'IEF', 'TLT']
bond_tickers = ['^MOVE']


#Data Fetching
try:
    curve_data = fetch_yahoo_data(curve_tickers)
    currency_data = fetch_yahoo_data(currency_tickers)
    credit_data = fetch_yahoo_data(credit_tickers)
    rates_data = fetch_yahoo_data(rates_tickers)
    bond_data = fetch_yahoo_data(bond_tickers)
except Exception as e:
    print(f"Error fetching data: {e}")


#Calculations and Alerts
M3 = curve_data['^IRX']['Close'].dropna()
Y2 = curve_data['2YY=F']['Close'].dropna()
Y5 = curve_data['^FVX']['Close'].dropna()
Y10 = curve_data['^TNX']['Close'].dropna()
Y30 = curve_data['^TYX']['Close'].dropna()

st.header("Rates Curve Analysis")
Y2Y10_spread =  Y10 - Y2
Y5Y30_spread = Y30 - Y5
M3Y10_spread = Y10 - M3
y2y10_rolling_z1 = (Y2Y10_spread - Y2Y10_spread.rolling(window=60).mean()) / Y2Y10_spread.rolling(window=60).std()
y2y10_rolling_z2 = (Y2Y10_spread - Y2Y10_spread.rolling(window=120).mean()) / Y2Y10_spread.rolling(window=120).std()
y5y30_rolling_z1 = (Y5Y30_spread - Y5Y30_spread.rolling(window=60).mean()) / Y5Y30_spread.rolling(window=60).std()
y5y30_rolling_z2 = (Y5Y30_spread - Y5Y30_spread.rolling(window=120).mean()) / Y5Y30_spread.rolling(window=120).std()
y10_rolling_z1 = (Y10 - Y10.rolling(window=60).mean()) / Y10.rolling(window=60).std()
rolling_z = pd.concat([y2y10_rolling_z1, y2y10_rolling_z2, y5y30_rolling_z1, y5y30_rolling_z2], axis=1)

rolling_z.columns = ['Y2Y10_Z60D', 'Y2Y10_Z120D', 'Y5Y30_Z60D', 'Y5Y30_Z120D']
fig = px.line(rolling_z, title="Rates Curve Spread Z-Scores")
st.plotly_chart(fig, use_container_width=True)
st.write(rolling_z.tail(6))

st.header("Curve Momentum and Reversal Risk")
roc_spread1_20 = Y2Y10_spread - Y2Y10_spread.shift(20)
roc_spread1_60 = Y2Y10_spread - Y2Y10_spread.shift(60)
roc_spread2_20 = Y5Y30_spread - Y5Y30_spread.shift(20)
roc_spread2_60 = Y5Y30_spread - Y5Y30_spread.shift(60)
roc_spread3_20 = M3Y10_spread - M3Y10_spread.shift(20)
roc_spread3_60 = M3Y10_spread - M3Y10_spread.shift(60)
roc_10Y_20 = Y10 - Y10.shift(20)
roc_10Y_60 = Y10 - Y10.shift(60)

roc = pd.concat([roc_spread1_20, roc_spread1_60, roc_spread2_20, roc_spread2_60,
           roc_spread3_20, roc_spread3_60, roc_10Y_20, roc_10Y_60], axis=1)
roc.columns = ['Y2Y10_ROC_20D', 'Y2Y10_ROC_60D', 'Y5Y30_ROC_20D', 'Y5Y30_ROC_60D',
               'M3Y10_ROC_20D', 'M3Y10_ROC_60D', '10Y_ROC_20D', '10Y_ROC_60D']
fig2 = px.line(roc, title="Rates Curve Momentum Indicators")
st.plotly_chart(fig2, use_container_width=True)

st.write(roc.tail(6))

acc_spread1 = roc_spread1_20.cumsum() - roc_10Y_20.cumsum()
acc_spread2 = roc_spread2_20.cumsum() - roc_10Y_20.cumsum()
acc_spread3 = roc_spread3_20.cumsum() - roc_10Y_20.cumsum()
acc_10Y = roc_10Y_20.cumsum() - roc_10Y_60.cumsum()

standardized_acc1 = (acc_spread1 - acc_spread1.rolling(window=120).mean()) / acc_spread1.rolling(window=120).std()
standardized_acc2 = (acc_spread2 - acc_spread2.rolling(window=120).mean()) / acc_spread2.rolling(window=120).std()
standardized_acc3 = (acc_spread3 - acc_spread3.rolling(window=120).mean()) / acc_spread3.rolling(window=120).std()
standardized_acc4 = (acc_10Y - acc_10Y.rolling(window=120).mean()) / acc_10Y.rolling(window=120).std()
acc_sum =  0.25*standardized_acc1 + 0.25*standardized_acc2 + 0.25*standardized_acc3 + 0.25*standardized_acc4

if acc_spread1.iloc[-1] or acc_spread2.iloc[-1] or acc_spread3.iloc[-1] or acc_10Y.iloc[-1] > 1:
    st.warning("Accelerating Move")
elif acc_spread1.iloc[-1] or acc_spread2.iloc[-1] or acc_spread3.iloc[-1] or acc_10Y.iloc[-1] < -1:
    st.warning("Decelerating Move")
elif abs(acc_sum.iloc[-1]) >= 1:
    st.info("Noise/Transition")

if acc_sum.iloc[-1] > 1:
    st.warning("Strong Acceleration Detected - Monitor Closely")
elif acc_sum.iloc[-1] < -1:
    st.warning("Strong Deceleration Detected - Monitor Closely")
elif abs(acc_sum.iloc[-1]) < 0.15:
    st.info("Market in Equilibrium - Low Stress")

fig3 = px.line(acc_sum, title="Rates Curve Acceleration Indicator")
st.plotly_chart(fig3, use_container_width=True)

acc_df = pd.DataFrame({
    'Spread1': standardized_acc1,
    'Spread2': standardized_acc2,
    'Spread3': standardized_acc3,
    '10Y': standardized_acc4,
    'CurveScore': acc_sum
})
rolling_acc = acc_df.rolling(window=5).median()


heatmap_data = rolling_acc[['Spread1','Spread2','Spread3','10Y']].T

fig = px.imshow(
    heatmap_data,
    labels=dict(x="Date", y="Series", color="Acceleration Z-Score"),
    x=heatmap_data.columns,
    y=heatmap_data.index,
    color_continuous_scale='RdYlGn_r',  
    aspect="auto"
)

fig.update_layout(
    title="Rolling Acceleration Heatmap",
    xaxis_title="Date",
    yaxis_title="Series",
    template='plotly_dark'  
)

st.plotly_chart(fig, use_container_width=True)

st.header("Regime Classification")
if y10_rolling_z1.iloc[-1] > 1 and y2y10_rolling_z1.iloc[-1] > 1 and y5y30_rolling_z1.iloc[-1] > 1 and m3y10_rolling_z1.iloc[-1] > 1 and acc_sum.iloc[-1] > 0.75:
    st.error("Broad curve sell-off is accelerating. Move is coherent across front, belly, and long end. Elevated risk of disorderly repricing.")
    st.warning("Reduce Duration, Avoid curve-carry trades, watch funding/volatility.")
elif y10_rolling_z1.iloc[-1] > 1 and y5y30_rolling_z1.iloc[-1] < -1 and m3y10_rolling_z1.iloc[-1] < -1 and acc_sum.iloc[-1] <= -0.25:
    st.error("Rates sell-off accelerating with front-end led flattening. Policy error risk rising.")
    st.warning("Hawkish repricing, curve flatteners favoured, growth downside risk")
elif y10_rolling_z1.iloc[-1] > 1 and y2y10_rolling_z1.iloc[-1] > 1 and y5y30_rolling_z1.iloc[-1] > 1 and abs(m3y10_rolling_z1.iloc[-1]) >= 0:
    st.error("Steepening is accelerating alongside rising yields. Term-premium driven move.")
    st.warning("Steepeners favoured, inflation risk priced in, equity correlation likely positive")
elif y10_rolling_z1.iloc[-1] < -1 and y2y10_rolling_z1.iloc[-1] < -1 and y5y30_rolling_z1.iloc[-1] < -1 and m3y10_rolling_z1.iloc[-1] < -1 and acc_sum.iloc[-1] <= -0.75:
    st.error("Rates rally is accelerating across the curve. Defensive regime emerging.")
    st.warning("Duration Bid, Risk-off confirmation, credit sensitivity rising")
elif abs(acc_sum.iloc[-1]) < 0.25:
    st.error("Acceleration signals mixed. Curve lacks coherence. Avoid directional conviction.")
    st.warning("Reduce Leverage, Favor relative value, expect mean reversion")
else:
    st.info("No clear regime signal detected. Monitor developments closely.")