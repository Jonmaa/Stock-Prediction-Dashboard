# Data Full Stack Methodology

This document describes the 11-step methodology used in this project.

## Overview

The **Data Full Stack** approach covers the complete lifecycle of a data science project, from data acquisition to deployment and documentation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA FULL STACK PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│  1. Data Acquisition  →  2. EDA  →  3. Cleaning                 │
│           ↓                                                      │
│  4. Feature Engineering  →  5. Modeling  →  6. Pipeline         │
│           ↓                                                      │
│  7. Validation  →  8. Optimization  →  9. Streamlit             │
│           ↓                                                      │
│  10. Deployment  →  11. Documentation                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Data Acquisition

**Notebook:** `01_data_acquisition.ipynb`

### Objective
Collect historical stock data for the Magnificent 7 tech stocks.

### Data Sources
- **yfinance**: Primary source for OHLCV data
- **Yahoo Finance API**: Company information

### Output
- `data/raw/{TICKER}_raw.csv`: Raw stock data for each ticker
- `data/raw/market_indices.csv`: Market index data
- `data/raw/company_info.csv`: Company metadata

### Key Code
```python
import yfinance as yf

ticker = yf.Ticker('AAPL')
df = ticker.history(period='10y')
df.to_csv('data/raw/AAPL_raw.csv')
```

---

## Step 2: Exploratory Data Analysis (EDA)

**Notebook:** `02_eda.ipynb`

### Objective
Understand data patterns, distributions, and relationships.

### Analysis Performed
1. **Descriptive Statistics**: Mean, std, min, max, quartiles
2. **Distribution Analysis**: Returns distribution, skewness, kurtosis
3. **Correlation Analysis**: Cross-stock correlations
4. **Seasonality Analysis**: Day-of-week, monthly patterns
5. **Volatility Analysis**: Rolling volatility over time

### Visualizations
- Correlation heatmap
- Returns distribution histograms
- Volatility time series
- Seasonality bar charts

---

## Step 3: Data Cleaning

**Notebook:** `03_cleaning.ipynb`

### Objective
Handle missing values, outliers, and data quality issues.

### Tasks
1. **Missing Values**: Forward fill, interpolation
2. **Outlier Detection**: Z-score > 3 flagging
3. **Data Validation**: Price sanity checks
4. **Date Alignment**: Ensure consistent trading days

### Output
- `data/cleaned/{TICKER}_cleaned.csv`
- `data/cleaned/cleaning_metadata.json`

---

## Step 4: Feature Engineering

**Notebook:** `04_feature_engineering.ipynb`

### Objective
Create predictive features from raw price data.

### Features Created (50+)

| Category | Features | Count |
|----------|----------|-------|
| Trend | SMA, EMA, MACD, ADX | 12 |
| Momentum | RSI, Stochastic, ROC | 8 |
| Volatility | Bollinger Bands, ATR | 6 |
| Volume | OBV, Volume Ratio | 3 |
| Lag Features | Return/RSI lags 1-20 | 12 |
| Rolling Stats | Mean, Std, Volatility | 9 |

### Target Variable
```python
df['Target'] = (df['close'].pct_change(1).shift(-1) > 0).astype(int)
# 1 = Price goes UP next day
# 0 = Price goes DOWN next day
```

---

## Step 5: Modeling

**Notebook:** `05_modeling.ipynb`

### Objective
Train baseline machine learning models.

### Models Tested
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. **LightGBM** ← Selected

### Why LightGBM?
- Fast training speed
- Low memory usage
- Good with high-dimensional data
- Handles categorical features
- Built-in regularization

---

## Step 6: Pipeline

**Notebook:** `06_pipeline.ipynb`

### Objective
Create production-ready ML pipelines.

### Pipeline Components
```
Raw Data → Technical Indicators → Lag Features → Rolling Stats
    ↓
Clean Data → Scale Features → Train Model → Save Artifacts
```

### Saved Artifacts
- `{ticker}_model.pkl`: Trained LightGBM model
- `{ticker}_scaler.pkl`: Fitted StandardScaler
- `{ticker}_features.json`: Feature column list
- `{ticker}_metadata.json`: Training metadata

---

## Step 7: Validation

**Notebook:** `07_validation.ipynb`

### Objective
Rigorously evaluate model performance.

### Validation Methods
1. **Time Series Cross-Validation**: 5-fold with temporal ordering
2. **Walk-Forward Validation**: Rolling window backtesting
3. **Statistical Significance**: Binomial test, t-test, Wilcoxon
4. **Stability Analysis**: Performance across market regimes

### Key Metrics
- Accuracy
- Precision / Recall / F1
- ROC-AUC
- Confusion Matrix

---

## Step 8: Optimization

**Notebook:** `08_optimization.ipynb`

### Objective
Tune hyperparameters to maximize performance.

### Method
RandomizedSearchCV with TimeSeriesSplit cross-validation.

### Parameter Grid
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'num_leaves': [15, 31, 63],
    'min_child_samples': [10, 20, 30]
}
```

### Results
- ~8% improvement over baseline
- Best model: LightGBM with optimized params

---

## Step 9: Streamlit Dashboard

**File:** `app/streamlit_app.py`

### Objective
Create an interactive web dashboard.

### Features
1. Stock selector (7 stocks)
2. Date range filter
3. Candlestick chart with indicators
4. Model predictions with confidence
5. Technical indicator analysis
6. Performance statistics

---

## Step 10: Deployment

**Directory:** `deploy/`

### Objective
Deploy the application for public access.

### Platform: Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Auto-deploy on push

---

## Step 11: Documentation

### Objective
Document everything for reproducibility.

### Documents
- `README.md`: Project overview
- `CHANGELOG.md`: Version history
- `CONTRIBUTING.md`: Contribution guide
- `docs/API.md`: API reference
- `docs/METHODOLOGY.md`: This document

---

## Lessons Learned

### Challenges
1. **Market Efficiency**: Stock prices are inherently hard to predict
2. **Overfitting**: Easy to overfit with many features
3. **Data Leakage**: Must be careful with time series data

### Key Insights
1. Simple models often perform better
2. Feature engineering is crucial
3. Proper validation is essential
4. ~51% accuracy is realistic for stock prediction

### Future Improvements
1. Add sentiment analysis from news
2. Include macroeconomic indicators
3. Implement ensemble methods
4. Add real-time data updates
