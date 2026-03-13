# Diabetes Prediction using Machine Learning

This project applies machine learning techniques to predict diabetes risk using a public diabetes dataset. The objective is to build and evaluate classification models that can identify whether a patient is likely to have diabetes based on medical attributes.

## Project Overview

In this project, two machine learning algorithms were implemented and compared:

- Random Forest Classifier
- Decision Tree Classifier

The models were trained on a diabetes dataset and evaluated using standard classification metrics to assess performance.

## Dataset

The dataset contains medical diagnostic measurements used to predict the presence of diabetes.  
Each row represents a patient record with features such as glucose level, BMI, age, and other health indicators.

Target variable:
- **Outcome**
  - 1 = Diabetes
  - 0 = No Diabetes

## Machine Learning Workflow

The following steps were performed in the project:

1. **Data Loading**
   - Imported the dataset using Python and Pandas.

2. **Data Preprocessing**
   - Converted categorical values to numerical values using label encoding.
   - Handled missing values using NumPy.

3. **Feature Selection**
   - Separated features (`X`) and target variable (`y`).

4. **Train-Test Split**
   - Split the dataset into training and testing sets (60% training, 40% testing).

5. **Model Training**
   - Implemented a Random Forest Classifier using Scikit-learn.
   - Trained the model on the training dataset.

6. **Model Evaluation**
   - Evaluated model performance using:
     - Accuracy
     - Confusion Matrix
     - Precision
     - Recall
     - F1 Score

## Results

Model performance results:

- **Train Accuracy:** 1.0  
- **Test Accuracy:** 0.75  

Evaluation metrics were calculated using confusion matrix and classification report to analyze model predictions.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## Project Structure

    diabetes-prediction-ml
    │
    ├── diabetes_prediction.py
    ├── diabetes_prediction.ipynb
    ├── diabetes.csv
    └── README.md

## Future Improvements

Possible improvements for this project include:

- Hyperparameter tuning for improved model performance
- Testing additional machine learning models
- Feature engineering to improve prediction accuracy
- Visualization of model performance metrics

## Author

Rungphailin Siamphupong
