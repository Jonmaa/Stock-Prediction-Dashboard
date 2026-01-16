"""
Tests for Stock Prediction Dashboard
"""

import pytest
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))


class TestImports:
    """Test that all required imports work"""
    
    def test_pandas_import(self):
        import pandas as pd
        assert pd is not None
    
    def test_numpy_import(self):
        import numpy as np
        assert np is not None
    
    def test_sklearn_import(self):
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        assert TimeSeriesSplit is not None
        assert StandardScaler is not None
    
    def test_lightgbm_import(self):
        from lightgbm import LGBMClassifier
        assert LGBMClassifier is not None
    
    def test_ta_import(self):
        import ta
        assert ta is not None
    
    def test_plotly_import(self):
        import plotly.graph_objects as go
        assert go is not None
    
    def test_streamlit_import(self):
        import streamlit as st
        assert st is not None


class TestDataLoading:
    """Test data loading functionality"""
    
    def test_data_directory_exists(self):
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        assert data_dir.exists(), "Data directory should exist"
    
    def test_csv_files_exist(self):
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        csv_files = list(data_dir.glob('*_raw.csv'))
        assert len(csv_files) > 0, "Should have CSV data files"


class TestModels:
    """Test model loading functionality"""
    
    def test_models_directory_exists(self):
        models_dir = Path(__file__).parent.parent / 'optimization' / 'models'
        assert models_dir.exists(), "Models directory should exist"
    
    def test_model_files_exist(self):
        models_dir = Path(__file__).parent.parent / 'optimization' / 'models'
        pkl_files = list(models_dir.glob('*.pkl'))
        assert len(pkl_files) > 0, "Should have model files"


class TestTechnicalIndicators:
    """Test technical indicator calculations"""
    
    def test_rsi_calculation(self):
        import pandas as pd
        import numpy as np
        import ta
        
        # Create sample data
        np.random.seed(42)
        close = pd.Series(np.random.randn(100).cumsum() + 100)
        
        # Calculate RSI
        rsi = ta.momentum.rsi(close, window=14)
        
        # RSI should be between 0 and 100
        assert rsi.dropna().min() >= 0
        assert rsi.dropna().max() <= 100
    
    def test_sma_calculation(self):
        import pandas as pd
        import numpy as np
        import ta
        
        # Create sample data
        close = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Calculate SMA
        sma = ta.trend.sma_indicator(close, window=3)
        
        # Check expected values
        assert sma.iloc[-1] == pytest.approx(9.0, rel=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
