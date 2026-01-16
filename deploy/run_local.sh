#!/bin/bash
# Stock Prediction Dashboard - Local Run Script (Linux/Mac)

echo "========================================"
echo " Stock Prediction Dashboard"
echo " Local Development Server"
echo "========================================"

# Activate virtual environment
source .venv/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Could not activate virtual environment"
    echo "Please run: python -m venv .venv"
    exit 1
fi

echo ""
echo "Virtual environment activated"
echo "Starting Streamlit server..."
echo ""

# Run Streamlit
streamlit run app/streamlit_app.py --server.port 8501
