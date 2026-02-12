@echo off
echo Starting Insurance Fraud Detection Dashboard...
cd /d "%~dp0"
streamlit run src/app.py
pause
