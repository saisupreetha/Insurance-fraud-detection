# Insurance Fraud Detector

Lightweight Streamlit app that uses machine learning and heuristic rules
to produce categorical fraud risk assessments and professional PDF reports
for insurance claims.

## Features
- Fast claim assessment using a pre-trained XGBoost model (default)
- Heuristic risk drivers combined with model output for robust decisions
- Inline AI assistant to answer basic questions about the analysis
- Audit-ready PDF report generation (in `reports/`)

## Quick Start

1. Create and activate a Python virtual environment.

Windows (PowerShell):

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run src/app.py
```

3. Open the URL printed by Streamlit (usually http://localhost:8501).

## Generating a PDF report
On the Result page click `Generate & Download PDF` to create a report
which will be saved under the `reports/` folder.

## Models
Pre-trained model files and encoders are expected in the `models/` folder:
- `xgboost.pkl`, `random_forest.pkl`, `logistic_regression.pkl`, `label_encoders.pkl`

## Notes
- If the app fails to start, ensure your virtual environment is active and
	dependencies installed via `requirements.txt`.
- Streamlit API differences between versions may affect `st.experimental_*`
	functions; the app uses `st.session_state` and `st.stop()` for navigation.

## Contact
For issues, open an issue in the project or contact the maintainer.

