# model.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def load_data(path):
    df = pd.read_csv(path)

    # handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    return df


def prepare_data(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    return X, y


def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return acc, cm, report


def feature_importance(model, X):
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=True)

    return importance


if __name__ == "__main__":
    df = load_data("data/diabetes.csv")

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    acc, cm, report = evaluate_model(model, X_test, y_test)

    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)