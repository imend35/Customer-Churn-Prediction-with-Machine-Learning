# Customer Churn Prediction with Machine Learning

## Project Overview

This project aims to predict customer churn in a telecom company using Machine Learning techniques. By analyzing customer demographics, service usage, and billing information, the goal is to identify customers who are at risk of leaving the service.

---

##  Business Problem

Customer churn is one of the biggest challenges for subscription-based businesses. Losing customers directly impacts revenue.

 The goal of this project is to:
- Predict which customers are likely to churn  
- Understand key factors driving churn  
- Provide actionable insights to reduce customer loss  

---

## Dataset

- Total records: **7043 customers**
- Features: **21 variables**
- Target: **Churn (Yes/No)**

### Key Features:
- Tenure (customer lifetime)
- MonthlyCharges
- Contract type
- Payment method
- Internet service type

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights from data analysis:

- Customers with **low tenure** have higher churn risk  
- Higher **Monthly Charges** increase churn probability  
- **Month-to-month contracts** show the highest churn  
- Customers using **Electronic Check** churn more  
- **Fiber optic users** have higher churn rates  

---

## Models Used

### Logistic Regression
- Simple and interpretable model  
- Higher accuracy  
- Lower recall (misses churn customers)

### Random Forest
- Captures complex patterns  
- Higher recall  
- Better at identifying churn customers  

---

## Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|---------|----------|--------|---------|
| Logistic Regression | ~0.80   | ~0.65    | ~0.57  | ~0.60   |
| Random Forest       | ~0.77   | ~0.55    | ~0.75  | ~0.63   |

---

## Key Insight

👉 In churn prediction, **Recall is more important than Accuracy**.

Random Forest outperformed Logistic Regression by:
- Reducing missed churn customers (False Negatives)
- Increasing detection of high-risk customers

---

## Business Impact

Based on the analysis:

### High-Risk Customers:
- Month-to-month contract  
- High monthly charges  
- Low tenure  
- Electronic check users  
- Fiber optic customers  

### Recommended Actions:
- Offer discounts for long-term contracts  
- Encourage auto-payment methods  
- Improve onboarding experience  
- Focus on high-value customer satisfaction  

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Seaborn, Matplotlib  
- Jupyter Notebook  

---

## Project Structure
churn-prediction/
│
├── data/
├── notebooks/
├── src/
├── README.md


---

## Future Improvements

- Hyperparameter tuning  
- Feature engineering  
- Deployment with Streamlit  
- Real-time prediction system  

---

## Author

**Esila Nur Demirci**  
Senior Business Analyst | AI & Data Science Enthusiast  

---

## Final Note

This project demonstrates how machine learning can be used not only to predict customer behavior but also to drive business decisions and improve customer retention strategies.
