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
from service import USING_CROSS_VALIDATION, label_columns,get_test_and_train_df,plot_confusion_matrix,merge_datasets,load_and_mark_potential_synthetic_data,load_and_split_dataset,ADD_SYNTHETIC_DATA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

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


def train_and_evaluate_models():
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


# def evaluate_models_with_stratified_kfold(df, n_splits=3):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     best_model_name = None
#     best_accuracy = 0
#     best_y_true = []
#     best_y_pred = []

#     # Extract features and labels from the DataFrame
#     X = df['Question'].to_numpy()  
#     y = df['Label'].to_numpy()   
#     is_synthetic = df['is_synthetic'].to_numpy() if ADD_SYNTHETIC_DATA else None

#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(X) 

#     for model_name, model in models.items():
#         print(f"\nModel: {model_name}")

#         cv_accuracies = []
#         y_cv_true = []
#         y_cv_pred = []
        
#         for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#             print(f"Fold {fold + 1}/{n_splits}")
            
#             # Filter out synthetic data from validation set
#             val_idx = [idx for idx in val_idx if not is_synthetic[idx]] if ADD_SYNTHETIC_DATA else val_idx
            
#             print(f"Train size: {len(train_idx)}, Validation size (real-world only): {len(val_idx)}")
            
#             # Split the data
#             X_train_fold, X_val_fold = X[train_idx], X[val_idx]
#             y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
#             # Train the model on the training fold
#             model.fit(X_train_fold, y_train_fold)
            
#             # Evaluate on the validation fold
#             y_val_pred = model.predict(X_val_fold)
#             fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
#             cv_accuracies.append(fold_accuracy)

#             # Collect true and predicted labels for the classification report
#             y_cv_true.extend(y_val_fold)
#             y_cv_pred.extend(y_val_pred)

#         # Calculate average accuracy for the model
#         avg_accuracy = np.mean(cv_accuracies)
#         print(f"Cross-Validation Accuracies: {cv_accuracies}")
#         print(f"Average Cross-Validation Accuracy: {avg_accuracy:.4f}")
#         print("Classification Report on Cross-Validation:")
#         print(classification_report(y_cv_true, y_cv_pred, target_names=label_columns, zero_division=0))
#         print("-" * 50)

#         # Update the best model if the current one has higher accuracy
#         if avg_accuracy > best_accuracy:
#             best_model_name = model_name
#             best_accuracy = avg_accuracy
#             best_y_true = y_cv_true
#             best_y_pred = y_cv_pred

#     # Plot the confusion matrix for the best model
#     if best_model_name:
        # print(f"\nBest Model: {best_model_name} with Average Accuracy: {best_accuracy:.4f}")
        # output_file = f'CM_Classical_{best_model_name}'
        # title = f'Confusion Matrix {best_model_name}'
        # plot_confusion_matrix(best_y_true, best_y_pred, label_columns, output_file, title)


def evaluate_models_with_stratified_kfold(df, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_model_name = None
    best_accuracy = 0
    best_y_true = []
    best_y_pred = []

    # Extract features and labels from the DataFrame
    X = df['Question'].to_numpy()  
    y = df['Label'].to_numpy()   
    is_synthetic = df['is_synthetic'].to_numpy() if ADD_SYNTHETIC_DATA else None

    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Convert the text data into numerical features
    X = vectorizer.fit_transform(X)  # Convert the 'Question' column to TF-IDF features

    for model_name, model in models.items():
        print(f"\nModel: {model_name}")

        cv_accuracies = []
        y_cv_true = []
        y_cv_pred = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{n_splits}")
            
            # Filter out synthetic data from validation set
            val_idx = [idx for idx in val_idx if not is_synthetic[idx]] if ADD_SYNTHETIC_DATA else val_idx
            
            print(f"Train size: {len(train_idx)}, Validation size (real-world only): {len(val_idx)}")
            
            # Split the data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train the model on the training fold
            model.fit(X_train_fold, y_train_fold)
            
            # Evaluate on the validation fold
            y_val_pred = model.predict(X_val_fold)
            
            # Filter predictions to include only real-world data
            y_val_true_real_world = y_val_fold
            y_val_pred_real_world = y_val_pred

            if ADD_SYNTHETIC_DATA:
                y_val_true_real_world = [y_val_fold[i] for i in range(len(y_val_fold)) if not is_synthetic[val_idx[i]]]
                y_val_pred_real_world = [y_val_pred[i] for i in range(len(y_val_pred)) if not is_synthetic[val_idx[i]]]
            
            fold_accuracy = accuracy_score(y_val_true_real_world, y_val_pred_real_world)
            cv_accuracies.append(fold_accuracy)

            # Collect true and predicted labels for the classification report (real-world only)
            y_cv_true.extend(y_val_true_real_world)
            y_cv_pred.extend(y_val_pred_real_world)
            
            # Print classification report for each fold
            print(f"Classification Report for Fold {fold + 1}:")
            print(classification_report(y_val_true_real_world, y_val_pred_real_world, target_names=label_columns, zero_division=0))

        # Calculate average accuracy for the model
        avg_accuracy = np.mean(cv_accuracies)
        print(f"Cross-Validation Accuracies: {cv_accuracies}")
        print(f"Average Cross-Validation Accuracy: {avg_accuracy:.4f}")
        # print("Classification Report on whole read world dataset:")
        # print(classification_report(y_cv_true, y_cv_pred, target_names=label_columns, zero_division=0))
        print("-" * 50)

        # Update the best model if the current one has higher accuracy
        if avg_accuracy > best_accuracy:
            best_model_name = model_name
            best_accuracy = avg_accuracy
            best_y_true = y_cv_true
            best_y_pred = y_cv_pred

    # Print the classification report for the best model after cross-validation
    if best_model_name:
        print(f"\nBest Model: {best_model_name} with Average Accuracy: {best_accuracy:.4f}")
        print("Classification Report for Best Model:")
        print(classification_report(best_y_true, best_y_pred, target_names=label_columns, zero_division=0))
    
        output_file = f'CM_Classical_{best_model_name}'
        title = f'Confusion Matrix {best_model_name}'
        plot_confusion_matrix(best_y_true, best_y_pred, label_columns, output_file, title)



# def evaluate_models_with_stratified_kfold(df, n_splits=3):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     best_model_name = None
#     best_accuracy = 0
#     best_y_true = []
#     best_y_pred = []

#     real_world_indices = (
#     {i for i, example in enumerate(df) if not example['is_synthetic']}
#     if ADD_SYNTHETIC_DATA
#     else set(range(len(df))))

#     for model_name, model in models.items():
#         print(f"\nModel: {model_name}")
#         # Perform StratifiedKFold Cross-Validation
#         cv_accuracies = []
#         y_cv_true = []
#         y_cv_pred = []
        
#         for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_tfidf, y_train)):
#             print(f"Fold {fold + 1}/{n_splits}")
#             print(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}")
            
#             # Split the data
#             X_train_fold = X_train_tfidf[train_idx]
#             X_val_fold = X_train_tfidf[val_idx]
#             y_train_fold = np.array(y_train)[train_idx]
#             y_val_fold = np.array(y_train)[val_idx]
            
#             # Train the model on the training fold
#             model.fit(X_train_fold, y_train_fold)
            
#             # Evaluate on the validation fold
#             y_val_pred = model.predict(X_val_fold)
#             fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
#             cv_accuracies.append(fold_accuracy)

#             # Collect true and predicted labels for the classification report
#             y_cv_true.extend(y_val_fold)
#             y_cv_pred.extend(y_val_pred)

#         # Calculate average accuracy for the model
#         avg_accuracy = np.mean(cv_accuracies)
#         print(f"Cross-Validation Accuracies: {cv_accuracies}")
#         print(f"Average Cross-Validation Accuracy: {avg_accuracy:.4f}")
#         print("Classification Report on Cross-Validation:")
#         print(classification_report(y_cv_true, y_cv_pred, target_names=label_columns, zero_division=0))
#         print("-" * 50)

#         # Update the best model if the current one has higher accuracy
#         if avg_accuracy > best_accuracy:
#             best_model_name = model_name
#             best_accuracy = avg_accuracy
#             best_y_true = y_cv_true
#             best_y_pred = y_cv_pred

#     # Plot the confusion matrix for the best model
#     if best_model_name:
#         print(f"\nBest Model: {best_model_name} with Average Accuracy: {best_accuracy:.4f}")
#         output_file = f'CM_Classical_{best_model_name}'
#         title = f'Confusion Matrix {best_model_name}'
#         plot_confusion_matrix(best_y_true, best_y_pred, label_columns, output_file, title)


# --------------- end: helper functions -----------------

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

if USING_CROSS_VALIDATION:
    dataset = load_and_mark_potential_synthetic_data()
    df = dataset.to_pandas()
    evaluate_models_with_stratified_kfold(df)
else:
    train_df, test_df =  get_test_and_train_df() 
    X_train = train_df['Question']
    y_train = train_df['Label']
    X_test = test_df['Question']
    y_test = test_df['Label']


    # Convert text data into TF-IDF features
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    train_and_evaluate_models()





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

