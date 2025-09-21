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

---

## 📸 Screenshots

### 1. Prediction Tab
<img src="screenshots/prediction.png" width="700">

### 2. Explainability (SHAP)
<img src="screenshots/shap.png" width="700">

### 3. Fairness Dashboard
<img src="screenshots/fairness.png" width="700">

---

##  Project Structure

