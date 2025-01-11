#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/nikkizhou/ML/blob/main/MT_Nikki.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from transformers import get_scheduler, AutoConfig, DataCollatorWithPadding,AutoTokenizer, AutoModelForSequenceClassification
from service import DEVICE, COMBINE_CATEGORIES, USING_CROSS_VALIDATION, MODEL_NAME, label_columns, model_name_simplified, compute_class_weights,tokenize_and_process_dataset,prepare_data_loaders, load_and_split_dataset,plot_confusion_matrix, load_and_mark_synthetic_data,train_and_evaluate_with_KFold,get_fold_string
from sklearn.model_selection import StratifiedKFold


# --------------- start: helper functions -----------------
def train_model(model, train_dataloader, train_dataset,eval_dataloader,num_epochs, gradient_accumulation_steps=4, lr=2e-5, weight_decay=0.01, early_stopping_patience=2):
    optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.to(DEVICE)

    # Compute and apply class weights
    train_dataset = train_dataset if train_dataset is not None else processed_datasets['train']
    class_weights = compute_class_weights(train_dataset).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_accuracy = 0
    epochs_no_improve = 0
    if not USING_CROSS_VALIDATION: num_epochs=1

    for epoch in range(num_epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            #outputs = model(**batch)
            outputs = model(**batch) 
            labels = batch["labels"] 
            logits = outputs.logits  

            loss = loss_fn(logits, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
    
        # Check Early Stopping Condition
        (accuracy,all_predictions,all_labels) = evaluate_model(model, eval_dataloader, label_columns)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}") 
                return

def evaluate_model(model, eval_dataloader, label_columns):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    
    return (accuracy,all_predictions,all_labels)

 
def print_classification_report(all_labels, all_predictions, label_columns, fold):
    print(f"Validation Classification Report {get_fold_string(fold)}:")

    # Find unique labels in predictions and actual labels
    predicted_labels = set(all_predictions)
    actual_labels = set(all_labels)
    expected_labels = set(range(len(label_columns)))

    # print(f"Unique predicted labels: {predicted_labels}")
    # print(f"Unique actual labels: {actual_labels}")
    # print(f"Expected labels based on label_columns: {expected_labels}")
    # print()

    # Adjust label set to the intersection of actual and expected labels
    label_indices = list(expected_labels & actual_labels)
    label_indices.sort()  # Ensure sorted order for consistency

    if len(label_indices) != len(label_columns):
        # Adjust target_names to match the labels being evaluated
        excluded_columns = [label_columns[i] for i in range(len(label_columns)) if i not in label_indices]
        print("Columns excluded in evaluation:", excluded_columns)
        label_columns = [label_columns[idx] for idx in label_indices]
      
    # Generate and print the classification report
    report = classification_report(
        all_labels, all_predictions, labels=label_indices, target_names=label_columns,zero_division=0
    )
    print(report)


def generate_output_and_title( fold):
    output_file = (
        f'CM_{model_name_simplified}_{get_fold_string(fold)}.png'
        if USING_CROSS_VALIDATION
        else f'CM_{model_name_simplified}.png'
    )
    title = (
        f'Confusion Matrix {model_name_simplified} {get_fold_string(fold)}'
        if USING_CROSS_VALIDATION
        else f'Confusion Matrix {model_name_simplified}'
    )
    return output_file, title


def train_and_evaluate_model(train_dataloader,eval_dataloader,fold,train_dataset=None):
    #print("Unique labels in dataset:", df['labels'].unique())
    train_model(model, train_dataloader, train_dataset,eval_dataloader, num_epochs=3, gradient_accumulation_steps=4)

    (accuracy,all_predictions,all_labels)  = evaluate_model(model, eval_dataloader, label_columns)
    if USING_CROSS_VALIDATION:
        print(f"Accuracy {get_fold_string(fold)}: {accuracy:.4f}")
    else:
        print(f"Accuracy: {accuracy:.4f}")
    print_classification_report(all_labels, all_predictions, label_columns,fold)

    output_file, title = generate_output_and_title(fold)
    plot_confusion_matrix(all_labels, all_predictions, label_columns,output_file, title)
    return accuracy

# --------------- end: helper functions ------------------

# 1. Load dataset
dataset=  load_and_mark_synthetic_data() if USING_CROSS_VALIDATION else load_and_split_dataset() 
dataset = dataset.rename_column('Label', 'labels')

# 2. Tokenize and process dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.add_special_tokens({'pad_token': '[PAD]'})

processed_datasets = tokenize_and_process_dataset(dataset,tokenizer)

# 3. Set Up the Optimizer and Learning Rate Scheduler
num_labels = 4 if COMBINE_CATEGORIES else 15

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.hidden_dropout_prob = 0.1  
model.config.attention_probs_dropout_prob = 0.1

torch.cuda.empty_cache()

# 4. Train and evaluate model
if USING_CROSS_VALIDATION:
    train_and_evaluate_with_KFold(processed_datasets,train_and_evaluate_model,tokenizer)
else:
    train_dataloader, eval_dataloader = prepare_data_loaders(processed_datasets, tokenizer)
    train_and_evaluate_model(train_dataloader,eval_dataloader, None)