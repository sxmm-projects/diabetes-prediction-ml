# Diabetes Prediction using Machine Learning

This project applies machine learning techniques to predict diabetes risk using a public diabetes dataset. The objective is to build and evaluate classification models that can identify whether a patient is likely to have diabetes based on medical attributes.

# 🩺 Diabetes Prediction AI

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/streamlit-1.22-orange?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 💡 Project Overview

This project is an interactive **Diabetes Prediction Dashboard** using **Random Forest Classifier**. It allows users to:

- Explore the dataset
- Train a machine learning model
- Visualize model performance
- Input patient data and get real-time predictions

The dashboard is built with **Streamlit** and provides a **user-friendly interface** for both data exploration and prediction.

---

## 📊 Features

- **Interactive Input Sidebar:** Enter patient metrics to predict diabetes risk.
- **Model Training:** Uses Random Forest with customizable parameters.
- **Performance Metrics:** Accuracy, confusion matrix, classification report.
- **Feature Importance Visualization:** See which medical indicators contribute most.
- **Real-time Prediction:** Predict diabetes risk with visual feedback (High Risk / Low Risk).

---

## 🛠 Technologies & Libraries

- Python 3.10+
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [streamlit](https://streamlit.io/)

---

## 📁 Project Structure

Diabetes-Prediction-AI/
├─ app.py              # Streamlit dashboard
├─ model.py            # ML model training & evaluation
├─ utils.py            # Helper functions for plotting & visualization
├─ data/
│   └─ diabetes.csv    # Dataset
├─ requirements.txt    # Python dependencies
└─ README.md           # Project documentation

---

## 🚀 Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/Diabetes-Prediction-AI.git
cd Diabetes-Prediction-AI

2. Create a virtual environment:
conda create -n diabetes_ai python=3.10
conda activate diabetes_ai

3. Install dependencies:
pip install -r requirements.txt

## ⚡ Run the App
streamlit run app.py
The dashboard will open in your browser.
Use the sidebar to enter patient data and predict diabetes risk.
Explore model metrics and feature importance.

🧠 How It Works
Data Loading: Reads diabetes.csv and cleans missing values.
Data Preparation: Splits features (X) and target (y).
Train/Test Split: 70% train, 30% test, stratified for balanced classes.
Model Training: Random Forest Classifier with 200 trees and max depth 6.
Evaluation: Computes accuracy, confusion matrix, and classification report.
Prediction: Accepts user input and predicts diabetes risk in real-time.

📈 Visualizations
Confusion Matrix
Feature Importance
Accuracy Metric
Prediction Badge (High Risk / Low Risk)

🏆 Future Improvements
Save and load trained model to speed up predictions
Add SHAP explanations for AI transparency
Deploy online for public access
Implement additional models (Logistic Regression, XGBoost)

👨‍💻 Author

Rungphailin Siamphupong
