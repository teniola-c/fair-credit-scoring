# Fair Credit Scoring Dashboard

An interactive machine learning app that predicts loan default probability, explains predictions using SHAP, and evaluates fairness across groups.

## Features
- Loan risk prediction
- SHAP explainability (local + global)
- Fairness metrics (demographic parity, equalized odds)
- Interactive Streamlit dashboard

##  Files
- `app.py` → main Streamlit dashboard
- `credit_model.pkl` → trained ML pipeline
- `processed_test.csv` → test data with sensitive features
- `requirements.txt` → dependencies

##  Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
