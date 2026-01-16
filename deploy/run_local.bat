@echo off
REM Stock Prediction Dashboard - Local Run Script (Windows)

echo ========================================
echo  Stock Prediction Dashboard
echo  Local Development Server
echo ========================================

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo ERROR: Could not activate virtual environment
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

echo.
echo Virtual environment activated
echo Starting Streamlit server...
echo.

REM Run Streamlit
streamlit run app/streamlit_app.py --server.port 8501

pause
