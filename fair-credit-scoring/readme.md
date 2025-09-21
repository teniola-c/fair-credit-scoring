# 💳 Fair & Explainable Credit Scoring Dashboard

## Project Overview
This project is an **end-to-end data analysis** of LendingClub loan data, with a focus on **risk prediction, transparency, and fairness**.  

The goal is not only to predict loan defaults, but also to:
- Explore and visualize borrower behavior
- Explain model predictions using **SHAP**
- Audit the model for **fairness across groups**
- Deliver results via an **interactive Streamlit dashboard**

## Why This Matters
Credit scoring directly impacts financial inclusion. Traditional models are often opaque and may introduce bias.  
This project demonstrates how a **data analyst** can combine:
- **Exploratory Data Analysis (EDA)** → patterns of defaults across loan types  
- **Predictive modeling** → simple ML pipeline to score applicants  
- **Explainability** → making “black box” models transparent with SHAP  
- **Fairness evaluation** → checking if approval rates differ across groups  
- **Dashboarding** → interactive app that communicates results clearly  

##  Tech Stack
- **Python**: pandas, numpy, matplotlib, seaborn  
- **Machine Learning**: scikit-learn, LightGBM  
- **Explainability**: SHAP  
- **Fairness**: Fairlearn  
- **Visualization/Dashboard**: Streamlit, Altair  

##  Features
1. **EDA**: Distribution of loan amounts, interest rates, default rates by grade/purpose  
2. **Prediction**: Enter applicant details → get default probability + decision  
3. **Explainability (SHAP)**: Why did the model make this prediction?  
4. **Fairness Dashboard**: Compare approval rates & metrics across groups (e.g. home ownership)  
5. **Bias Mitigation**: Simple per-group thresholds to reduce unfairness  

##  Project Structure
fair-credit-scoring/
│── app.py # Streamlit app
│── credit_model.pkl # trained ML pipeline
│── processed_test.csv # test data with labels + sensitive features
│── requirements.txt # dependencies
│── README.md # project documentation

## Run Locally
Clone the repo and install dependencies:
```bash
git clone https://github.com/yourusername/fair-credit-scoring.git
cd fair-credit-scoring
pip install -r requirements.txt

## Run the dashboard:
streamlit run app.py

