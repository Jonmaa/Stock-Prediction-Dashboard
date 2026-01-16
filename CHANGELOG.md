# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-16

### Added
- Initial release of Stock Prediction Dashboard
- Complete Data Full Stack methodology implementation (11 steps)
- Interactive Streamlit dashboard with:
  - Candlestick charts with technical indicators
  - Model predictions with confidence gauge
  - Technical analysis tabs (Momentum, Trend, Volatility)
  - Performance statistics
- LightGBM models for 7 Magnificent 7 stocks
- 50+ engineered features including:
  - Trend indicators (SMA, EMA, MACD, ADX)
  - Momentum indicators (RSI, Stochastic, ROC)
  - Volatility indicators (Bollinger Bands, ATR)
  - Volume indicators (OBV, Volume Ratio)
  - Lag features and rolling statistics
- Hyperparameter optimization with RandomizedSearchCV
- Cross-validation and walk-forward validation
- Statistical significance testing
- Deployment configuration for Streamlit Cloud

### Technical Details
- Python 3.11+ compatibility
- scikit-learn 1.3+ for ML pipeline
- LightGBM 4.0+ for gradient boosting
- Streamlit 1.28+ for web interface
- Plotly 5.15+ for interactive visualizations

## [Unreleased]

### Planned
- Real-time data updates
- Portfolio optimization features
- Additional ML models (XGBoost, Neural Networks)
- Sentiment analysis integration
- Alert system for predictions
