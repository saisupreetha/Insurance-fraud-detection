# Insurance Fraud Detection System 🕵️‍♂️🚫

## Overview
A Generative AI-powered system designed to detect and explain insurance fraud patterns. This project includes a machine learning pipeline for fraud classification, an interactive analyst dashboard, and an automated reporting system.

## Key Features
- **Fraud Detection Engine**: Utilizes XGBoost and Random Forest to classify claims.
- **AI Risk Explanation**: Generates natural language explanations for high-risk claims.
- **Analyst Dashboard**: A modern UI for reviewing claims and insights.
- **Chatbot Assistant**: A fraud analyst copilot for ad-hoc queries.
- **Auto-Reporting**: Generates PDF risk summary reports.

## Project Structure
```
├── data/               # Dataset storage
├── src/                # Source code
│   ├── model_train.py  # ML Training Pipeline
│   ├── app.py          # Dashboard Application
│   └── utils.py        # Helper functions
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training pipeline:
   ```bash
   python src/model_train.py
   ```
3. Launch the dashboard:
   ```bash
   streamlit run src/app.py
   ```
