#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from service import USING_CROSS_VALIDATION, label_columns,get_test_and_train_df,plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report

# --------------- start: helper functions -----------------
# def filter_labels_with_min_samples(min_samples=5):
#     # Get counts of each label
#     label_counts = y.value_counts()
   
#     # Identify labels with at least the minimum number of samples
#     labels_to_keep = label_counts[label_counts >= min_samples].index

#     # Filter X and y to keep only labels with sufficient samples
#     X_filtered = X[y.isin(labels_to_keep)]
#     y_filtered = y[y.isin(labels_to_keep)]

#     return X_filtered, y_filtered


def tune_model(model, X_train_tfidf, y_train):
    if isinstance(model, LogisticRegression):
        param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
    elif isinstance(model, SVC):
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    elif isinstance(model, RandomForestClassifier):
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}

    # Randomized Search for hyperparameter tuning
    random_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, verbose=1, random_state=42, n_jobs=-1)
    random_search.fit(X_train_tfidf, y_train)
    
    print(f"Best parameters for {model.__class__.__name__}: {random_search.best_params_}")
    return random_search.best_estimator_


def evaluate_models():
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        # Evaluate model
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_columns, zero_division=0))
        print("-" * 50)
        output_file = 'CM_Classical_' + name
        title = f'Confusion Matrix {name}'  
        plot_confusion_matrix(y_test, y_pred, label_columns, output_file, title)


# def evaluate_models_with_cross_validation( n_splits=5):
#     # Convert text data into TF-IDF features for the entire dataset
#     tfidf = TfidfVectorizer()
#     X_tfidf = tfidf.fit_transform(X)

#     # Perform cross-validation for each model
#     for model_name, model in models.items():
#         print(f"\nModel: {model_name}")

#         # Perform cross-validation and calculate accuracy for each fold
#         accuracies = cross_val_score(model, X_tfidf, y, cv=n_splits, scoring='accuracy')
#         print(f"Accuracies for each fold: {accuracies}")
#         print(f"Average Accuracy: {np.mean(accuracies):.4f}")

#         # Perform cross-validation to get predictions for each fold
#         y_pred = cross_val_predict(model, X_tfidf, y, cv=n_splits)

#         # Print the overall classification report
#         print(classification_report(y, y_pred, target_names=label_columns, zero_division=0)) 
#         print("-" * 50)

def evaluate_models_with_cross_validation(n_splits=3):
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")

        # Evaluate on the predefined train-test split
        model.fit(X_train_tfidf, y_train)
        y_test_pred = model.predict(X_test_tfidf)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"Test Set Accuracy: {test_accuracy:.4f}")
        print("Classification Report on Test Set:")
        print(classification_report(y_test, y_test_pred, target_names=label_columns, zero_division=0))
        print("-" * 50)

        # Perform cross-validation on the training data
        cv_accuracies = cross_val_score(model, X_train_tfidf, y_train, cv=n_splits, scoring='accuracy')
        y_cv_pred = cross_val_predict(model, X_train_tfidf, y_train, cv=n_splits)

        # Print cross-validation results
        print(f"Cross-Validation Accuracies: {cv_accuracies}")
        print(f"Average Cross-Validation Accuracy: {np.mean(cv_accuracies):.4f}")
        print("Classification Report on Cross-Validation:")
        print(classification_report(y_train, y_cv_pred, target_names=label_columns, zero_division=0))
        print("-" * 50)

        output_file = 'CM_Classical_' + model_name
        title = f'Confusion Matrix {model_name}'  
        plot_confusion_matrix(y_test, y_test_pred, label_columns, output_file, title)
# --------------- end: helper functions -----------------

# 1. Convert excel file to DataFrame
train_df, test_df = get_test_and_train_df()

X_train = train_df['Question']
y_train = train_df['Label']
X_test = test_df['Question']
y_test = test_df['Label']


# 2. Split the dataset into training and testing sets
# X = df['Question']  
# y = df['Label'] 

# X, y = filter_labels_with_min_samples()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Convert text data into TF-IDF features
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}


# tuned_models = {}
# for name, model in models.items():
#     print(f"Tuning model: {name}")
#     tuned_models[name] = tune_model(model, X_train_tfidf, y_train)

# Evaluate the best models with the tuned hyperparameters
# for name, model in tuned_models.items():
#     y_pred = model.predict(X_test_tfidf)
#     print(f"\nModel: {name}")
#     print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred, target_names=label_columns, zero_division=0))
#     print("-" * 50)


#3. Train and Evaluate models
if USING_CROSS_VALIDATION:
    evaluate_models_with_cross_validation()
else:
    evaluate_models()
