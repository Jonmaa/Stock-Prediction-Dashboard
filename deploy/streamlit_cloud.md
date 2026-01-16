# Streamlit Cloud Deployment Guide

## Prerequisites
1. GitHub account
2. Streamlit Cloud account (https://streamlit.io/cloud)

## Deployment Steps

### 1. Prepare Repository
```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Stock Prediction Dashboard"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/stock-prediction-dashboard.git

# Push
git push -u origin main
```

### 2. Connect to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your GitHub repository
4. Set main file path: `app/streamlit_app.py`
5. Click "Deploy"

### 3. Configuration

Create `.streamlit/secrets.toml` for any secrets (API keys, etc.):
```toml
# Example secrets
[api_keys]
yfinance_key = "your_api_key"
```

### 4. Environment Variables

In Streamlit Cloud dashboard, add environment variables:
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`

## File Structure Required
```
stock-prediction-dashboard/
├── app/
│   └── streamlit_app.py
├── data/
│   └── raw/
│       └── *.csv
├── optimization/
│   └── models/
│       └── *.pkl
├── requirements.txt
└── .streamlit/
    └── config.toml
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Check requirements.txt
2. **FileNotFoundError**: Verify data paths use relative paths
3. **Memory Error**: Reduce data size or upgrade Streamlit Cloud plan

### Resource Limits (Free Tier)
- RAM: 1GB
- Storage: Limited
- Sleep after inactivity
