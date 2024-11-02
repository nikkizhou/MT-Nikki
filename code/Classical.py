#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from service import USING_CROSS_VALIDATION, label_columns,process_excel_file

# --------------- start: helper functions -----------------
def filter_labels_with_min_samples(min_samples=5):
    # Get counts of each label
    label_counts = y.value_counts()
   
    # Identify labels with at least the minimum number of samples
    labels_to_keep = label_counts[label_counts >= min_samples].index

    # Filter X and y to keep only labels with sufficient samples
    X_filtered = X[y.isin(labels_to_keep)]
    y_filtered = y[y.isin(labels_to_keep)]

    return X_filtered, y_filtered


def evaluate_models():
    # Convert text data into TF-IDF features
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        # Evaluate model
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_columns, zero_division=0))
        print("-" * 50)


def evaluate_models_with_cross_validation( n_splits=5):
    # Convert text data into TF-IDF features for the entire dataset
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X)

    # Perform cross-validation for each model
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")

        # Perform cross-validation and calculate accuracy for each fold
        accuracies = cross_val_score(model, X_tfidf, y, cv=n_splits, scoring='accuracy')
        print(f"Accuracies for each fold: {accuracies}")
        print(f"Average Accuracy: {np.mean(accuracies):.4f}")

        # Perform cross-validation to get predictions for each fold
        y_pred = cross_val_predict(model, X_tfidf, y, cv=n_splits)

        # Print the overall classification report
        print(classification_report(y, y_pred, target_names=label_columns, zero_division=0)) 
        print("-" * 50)

# --------------- end: helper functions -----------------

# 1. Convert excel file to DataFrame
df = process_excel_file()

# 2. Split the dataset into training and testing sets
X = df['Question']  
y = df['labels'] 
X, y = filter_labels_with_min_samples()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# 3. Train and Evaluate models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

if USING_CROSS_VALIDATION:
    evaluate_models_with_cross_validation()
else:
    evaluate_models()


