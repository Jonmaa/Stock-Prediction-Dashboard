# API Reference

## Streamlit Dashboard (`app/streamlit_app.py`)

### Data Loading Functions

#### `load_stock_data(ticker: str) -> pd.DataFrame`
Load stock data from CSV file.

**Parameters:**
- `ticker`: Stock symbol (e.g., 'AAPL', 'MSFT')

**Returns:**
- DataFrame with OHLCV data indexed by date

**Example:**
```python
df = load_stock_data('AAPL')
print(df.head())
```

---

#### `load_model(ticker: str) -> Tuple[LGBMClassifier, StandardScaler]`
Load trained model and scaler for a specific stock.

**Parameters:**
- `ticker`: Stock symbol

**Returns:**
- Tuple of (model, scaler) or (None, None) if not found

---

#### `load_feature_cols() -> List[str]`
Load the list of feature column names.

**Returns:**
- List of feature column names used by the model

---

### Feature Engineering

#### `add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame`
Add technical indicators to a stock DataFrame.

**Parameters:**
- `df`: DataFrame with OHLCV columns

**Returns:**
- DataFrame with additional technical indicator columns

**Indicators Added:**
| Category | Indicators |
|----------|------------|
| Trend | SMA_10, SMA_20, SMA_50, EMA_10, EMA_20, MACD, ADX |
| Momentum | RSI, RSI_Fast, Stoch_K, Stoch_D, ROC_5, ROC_10 |
| Volatility | BB_High, BB_Low, BB_Width, ATR |
| Volume | OBV, Volume_SMA_20, Volume_Ratio |
| Returns | Return, Return_Lag_1..20, Return_Mean_5D..20D |

---

### Prediction

#### `make_prediction(df, model, scaler, feature_cols) -> Tuple[int, np.array]`
Make a prediction for the latest data point.

**Parameters:**
- `df`: DataFrame with features
- `model`: Trained classifier
- `scaler`: Fitted StandardScaler
- `feature_cols`: List of feature column names

**Returns:**
- Tuple of (prediction, probability_array)
  - prediction: 0 (DOWN) or 1 (UP)
  - probability_array: [prob_down, prob_up]

---

## Pipeline Classes (`notebooks/06_pipeline.ipynb`)

### `TechnicalIndicatorTransformer`

Transformer class for adding technical indicators.

```python
class TechnicalIndicatorTransformer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        pass
```

### `LagFeatureTransformer`

Transformer for creating lag features.

```python
class LagFeatureTransformer:
    def __init__(self, columns: List[str], lags: List[int]):
        pass
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
```

### `StockPredictionPipeline`

Full ML pipeline for stock prediction.

```python
class StockPredictionPipeline:
    def __init__(self, ticker: str):
        pass
    
    def fit(self, df: pd.DataFrame) -> 'StockPredictionPipeline':
        """Train the pipeline"""
        pass
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, np.array]:
        """Make prediction"""
        pass
    
    def save(self, directory: Path) -> None:
        """Save pipeline to disk"""
        pass
    
    @classmethod
    def load(cls, directory: Path) -> 'StockPredictionPipeline':
        """Load pipeline from disk"""
        pass
```

---

## Model Configuration

### LightGBM Optimized Parameters

```python
OPTIMIZED_PARAMS = {
    'num_leaves': 15,
    'n_estimators': 50,
    'min_child_samples': 20,
    'max_depth': 3,
    'learning_rate': 0.01,
    'random_state': 42,
    'verbosity': -1
}
```

### Feature Columns

Features are automatically determined from the data. Excluded columns:
```python
EXCLUDE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 
    'adj close', 'Target', 'date', 'Date', 'ticker', 'Ticker'
]
```

---

## Constants

### Tickers
```python
TICKERS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL']
```

### Stock Names
```python
STOCK_NAMES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla, Inc.',
    'AMZN': 'Amazon.com, Inc.',
    'META': 'Meta Platforms, Inc.',
    'GOOGL': 'Alphabet Inc.'
}
```

### Paths
```python
DATA_RAW = Path('../data/raw')
MODELS_DIR = Path('../optimization/models')
```
