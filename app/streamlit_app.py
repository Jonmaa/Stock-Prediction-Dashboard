"""
Stock Prediction Dashboard - Magnificent 7
Step 9: Streamlit Interactive Dashboard

This dashboard provides:
1. Stock price visualization
2. Technical indicators analysis
3. Model predictions
4. Performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
import json
from datetime import datetime, timedelta
import ta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / 'data' / 'raw'
OPTIMIZATION_DIR = BASE_DIR / 'optimization'
MODELS_DIR = OPTIMIZATION_DIR / 'models'

TICKERS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL']

STOCK_NAMES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla, Inc.',
    'AMZN': 'Amazon.com, Inc.',
    'META': 'Meta Platforms, Inc.',
    'GOOGL': 'Alphabet Inc.'
}

# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data
def load_stock_data(ticker):
    """Load stock data from CSV"""
    file_path = DATA_RAW / f'{ticker}_raw.csv'
    if file_path.exists():
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df.columns = df.columns.str.lower()
        return df
    return None

@st.cache_resource
def load_model(ticker):
    """Load trained model and scaler"""
    model_path = MODELS_DIR / f'{ticker}_optimized_model.pkl'
    scaler_path = MODELS_DIR / f'{ticker}_scaler.pkl'
    
    if model_path.exists() and scaler_path.exists():
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

@st.cache_data
def load_feature_cols():
    """Load feature columns"""
    feature_path = MODELS_DIR / 'feature_cols.json'
    if feature_path.exists():
        with open(feature_path, 'r') as f:
            return json.load(f)
    return None

def add_technical_indicators(df):
    """Add technical indicators to dataframe"""
    df = df.copy()
    
    # Trend indicators
    df['SMA_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['EMA_10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # ADX
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    df['ADX'] = adx.adx()
    df['ADX_Pos'] = adx.adx_pos()
    df['ADX_Neg'] = adx.adx_neg()
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['RSI_Fast'] = ta.momentum.rsi(df['close'], window=7)
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # ROC
    df['ROC_5'] = ta.momentum.roc(df['close'], window=5)
    df['ROC_10'] = ta.momentum.roc(df['close'], window=10)
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
    df['BB_Pct'] = bb.bollinger_pband()
    
    # ATR
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    
    # Volume indicators
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['Volume_SMA_20'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA_20']
    
    # Returns and lags
    df['Return'] = df['close'].pct_change()
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
        if 'RSI' in df.columns:
            df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
    
    # Rolling stats
    for window in [5, 10, 20]:
        df[f'Return_Mean_{window}D'] = df['Return'].rolling(window).mean()
        df[f'Return_Std_{window}D'] = df['Return'].rolling(window).std()
        df[f'Volatility_{window}D'] = df['Return'].rolling(window).std() * np.sqrt(252)
    
    return df

def make_prediction(df, model, scaler, feature_cols):
    """Make prediction for the latest data point"""
    df_features = df.dropna(subset=feature_cols)
    if len(df_features) == 0:
        return None, None
    
    X = df_features[feature_cols].iloc[-1:].values
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    return prediction, probability

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.title("📈 Stock Prediction")
st.sidebar.markdown("---")

# Stock selection
selected_ticker = st.sidebar.selectbox(
    "Select Stock",
    TICKERS,
    format_func=lambda x: f"{x} - {STOCK_NAMES[x]}"
)

# Date range
st.sidebar.markdown("### 📅 Date Range")
date_options = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730,
    "All Data": None
}
selected_range = st.sidebar.selectbox("Select Period", list(date_options.keys()), index=3)

# Technical indicators toggle
st.sidebar.markdown("### 📊 Technical Indicators")
show_sma = st.sidebar.checkbox("Show SMA (10, 20, 50)", value=True)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=False)
show_volume = st.sidebar.checkbox("Show Volume", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This dashboard uses LightGBM models optimized for stock price direction prediction. "
    "Models were trained on historical data with 50+ technical features."
)

# ============================================================================
# Main Content
# ============================================================================

# Load data
df = load_stock_data(selected_ticker)

if df is None:
    st.error(f"Could not load data for {selected_ticker}")
    st.stop()

# Add technical indicators
df = add_technical_indicators(df)

# Filter by date range
if date_options[selected_range] is not None:
    cutoff_date = df.index.max() - timedelta(days=date_options[selected_range])
    df_display = df[df.index >= cutoff_date].copy()
else:
    df_display = df.copy()

# Header
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Price",
        f"${df_display['close'].iloc[-1]:.2f}",
        f"{df_display['Return'].iloc[-1]*100:.2f}%"
    )

with col2:
    high_52w = df_display['high'].max()
    st.metric("52W High", f"${high_52w:.2f}")

with col3:
    low_52w = df_display['low'].min()
    st.metric("52W Low", f"${low_52w:.2f}")

with col4:
    avg_volume = df_display['volume'].mean()
    st.metric("Avg Volume", f"{avg_volume/1e6:.1f}M")

st.markdown("---")

# ============================================================================
# Price Chart
# ============================================================================

st.subheader(f"📈 {selected_ticker} - {STOCK_NAMES[selected_ticker]}")

# Create price chart
fig = make_subplots(
    rows=2 if show_volume else 1,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    row_heights=[0.7, 0.3] if show_volume else [1]
)

# Candlestick chart
fig.add_trace(
    go.Candlestick(
        x=df_display.index,
        open=df_display['open'],
        high=df_display['high'],
        low=df_display['low'],
        close=df_display['close'],
        name="OHLC"
    ),
    row=1, col=1
)

# SMA lines
if show_sma:
    for sma, color in [('SMA_10', 'orange'), ('SMA_20', 'blue'), ('SMA_50', 'purple')]:
        if sma in df_display.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_display.index,
                    y=df_display[sma],
                    mode='lines',
                    name=sma,
                    line=dict(color=color, width=1)
                ),
                row=1, col=1
            )

# Bollinger Bands
if show_bb:
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display['BB_High'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display['BB_Low'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.1)'
        ),
        row=1, col=1
    )

# Volume
if show_volume:
    colors = ['green' if c >= o else 'red' 
              for c, o in zip(df_display['close'], df_display['open'])]
    fig.add_trace(
        go.Bar(
            x=df_display.index,
            y=df_display['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# Prediction Section
# ============================================================================

st.markdown("---")
st.subheader("🤖 Model Prediction")

# Load model
model, scaler = load_model(selected_ticker)
feature_cols = load_feature_cols()

if model is not None and scaler is not None and feature_cols is not None:
    # Make prediction
    prediction, probability = make_prediction(df, model, scaler, feature_cols)
    
    if prediction is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.success("### 📈 Prediction: UP")
                st.markdown(f"**Confidence:** {probability[1]*100:.1f}%")
            else:
                st.error("### 📉 Prediction: DOWN")
                st.markdown(f"**Confidence:** {probability[0]*100:.1f}%")
        
        with col2:
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[1] * 100,
                title={'text': "Bullish Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green" if probability[1] > 0.5 else "red"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col3:
            st.markdown("### 📊 Model Info")
            st.markdown(f"- **Model:** LightGBM (Optimized)")
            st.markdown(f"- **Features:** {len(feature_cols)}")
            st.markdown(f"- **Last Update:** {df.index[-1].strftime('%Y-%m-%d')}")
    else:
        st.warning("Not enough data to make prediction")
else:
    st.warning("Model not found. Please run optimization first.")

# ============================================================================
# Technical Indicators Section
# ============================================================================

st.markdown("---")
st.subheader("📊 Technical Indicators")

tab1, tab2, tab3 = st.tabs(["Momentum", "Trend", "Volatility"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI Chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df_display.index, y=df_display['RSI'],
            mode='lines', name='RSI (14)',
            line=dict(color='purple')
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(title="RSI (14)", height=300, yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        # Stochastic
        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(
            x=df_display.index, y=df_display['Stoch_K'],
            mode='lines', name='%K',
            line=dict(color='blue')
        ))
        fig_stoch.add_trace(go.Scatter(
            x=df_display.index, y=df_display['Stoch_D'],
            mode='lines', name='%D',
            line=dict(color='orange')
        ))
        fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
        fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
        fig_stoch.update_layout(title="Stochastic Oscillator", height=300, yaxis_range=[0, 100])
        st.plotly_chart(fig_stoch, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df_display.index, y=df_display['MACD'],
            mode='lines', name='MACD',
            line=dict(color='blue')
        ))
        fig_macd.add_trace(go.Scatter(
            x=df_display.index, y=df_display['MACD_Signal'],
            mode='lines', name='Signal',
            line=dict(color='orange')
        ))
        colors = ['green' if v >= 0 else 'red' for v in df_display['MACD_Hist']]
        fig_macd.add_trace(go.Bar(
            x=df_display.index, y=df_display['MACD_Hist'],
            name='Histogram', marker_color=colors, opacity=0.5
        ))
        fig_macd.update_layout(title="MACD", height=300)
        st.plotly_chart(fig_macd, use_container_width=True)
    
    with col2:
        # ADX
        fig_adx = go.Figure()
        fig_adx.add_trace(go.Scatter(
            x=df_display.index, y=df_display['ADX'],
            mode='lines', name='ADX',
            line=dict(color='black', width=2)
        ))
        fig_adx.add_trace(go.Scatter(
            x=df_display.index, y=df_display['ADX_Pos'],
            mode='lines', name='+DI',
            line=dict(color='green')
        ))
        fig_adx.add_trace(go.Scatter(
            x=df_display.index, y=df_display['ADX_Neg'],
            mode='lines', name='-DI',
            line=dict(color='red')
        ))
        fig_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Trend Strength")
        fig_adx.update_layout(title="ADX (Trend Strength)", height=300)
        st.plotly_chart(fig_adx, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # ATR
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(
            x=df_display.index, y=df_display['ATR'],
            mode='lines', name='ATR',
            line=dict(color='blue')
        ))
        fig_atr.update_layout(title="Average True Range (ATR)", height=300)
        st.plotly_chart(fig_atr, use_container_width=True)
    
    with col2:
        # Volatility
        fig_vol = go.Figure()
        for window, color in [(5, 'green'), (10, 'blue'), (20, 'red')]:
            fig_vol.add_trace(go.Scatter(
                x=df_display.index, y=df_display[f'Volatility_{window}D'] * 100,
                mode='lines', name=f'{window}D',
                line=dict(color=color)
            ))
        fig_vol.update_layout(title="Annualized Volatility (%)", height=300)
        st.plotly_chart(fig_vol, use_container_width=True)

# ============================================================================
# Statistics Section
# ============================================================================

st.markdown("---")
st.subheader("📈 Performance Statistics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Returns Summary")
    returns = df_display['Return'].dropna()
    
    stats_df = pd.DataFrame({
        'Metric': ['Mean Daily Return', 'Std Daily Return', 'Annualized Return', 
                   'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown'],
        'Value': [
            f"{returns.mean()*100:.3f}%",
            f"{returns.std()*100:.3f}%",
            f"{returns.mean()*252*100:.2f}%",
            f"{returns.std()*np.sqrt(252)*100:.2f}%",
            f"{(returns.mean()/returns.std())*np.sqrt(252):.2f}",
            f"{((df_display['close']/df_display['close'].cummax())-1).min()*100:.2f}%"
        ]
    })
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

with col2:
    st.markdown("### Returns Distribution")
    fig_hist = px.histogram(
        returns * 100, 
        nbins=50,
        title="Daily Returns Distribution (%)"
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
    fig_hist.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>📊 Stock Prediction Dashboard | Data Full Stack Project</p>
        <p>⚠️ This is for educational purposes only. Not financial advice.</p>
    </div>
    """,
    unsafe_allow_html=True
)
