#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/nikkizhou/ML/blob/main/MT_Nikki.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report

from transformers import get_scheduler, AutoConfig,DataCollatorWithPadding,AutoTokenizer, AutoModelForSequenceClassification
from service import DEVICE, COMBINE_CATEGORIES,USING_CROSS_VALIDATION,MODEL_NAME,label_columns,compute_class_weights,tokenize_and_process_dataset,prepare_data_loaders, load_my_dataset
from sklearn.model_selection import StratifiedKFold

# --------------- start: helper functions -----------------
def train_model(model, train_dataloader, num_epochs, gradient_accumulation_steps=4, lr=2e-5, weight_decay=0.01, early_stopping_patience=2):
    optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.to(DEVICE)

    # Compute and apply class weights
    class_weights = compute_class_weights(processed_datasets['train']).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_accuracy = 0
    epochs_no_improve = 0

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

 
def print_classification_report(all_labels, all_predictions, label_columns):
    print("\nClassification Report:")

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


def train_and_evaluate_model(train_dataloader,eval_dataloader):
    #print("Unique labels in dataset:", df['labels'].unique())
    train_model(model, train_dataloader, num_epochs=3, gradient_accumulation_steps=4)

    (accuracy,all_predictions,all_labels)  = evaluate_model(model, eval_dataloader, label_columns)
    print(f"Accuracy: {accuracy:.4f}")
    print_classification_report(all_labels, all_predictions, label_columns)

    return accuracy

def prepare_data_loaders_for_kfold(dataset, tokenizer, batch_size=4):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    
    return dataloader

def train_and_evaluate_with_KFold():
    #kf = KFold(n_splits=3, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    full_dataset = processed_datasets['train']

    # Extract inputs and labels
    inputs = full_dataset.remove_columns('labels')
    labels = full_dataset['labels']

    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(inputs, labels)):
        print(f"\nFold {fold + 1}/{skf.n_splits}")

        # Create train and validation datasets
        train_inputs = inputs.select(train_idx)
        val_inputs = inputs.select(val_idx)

        # Use the indices to select labels for the respective splits
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        # Add labels to datasets
        train_dataset = train_inputs.add_column("labels", train_labels)
        val_dataset = val_inputs.add_column("labels", val_labels)

        # Prepare data loaders directly from the newly created datasets
        train_dataloader_kFold = prepare_data_loaders_for_kfold(train_dataset, tokenizer)
        eval_dataloader_kFold = prepare_data_loaders_for_kfold( val_dataset, tokenizer)

        accuracy = train_and_evaluate_model(train_dataloader_kFold, eval_dataloader_kFold)
        accuracies.append(accuracy)  

    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"\nAverage Accuracy across all folds: {average_accuracy:.4f}")

# --------------- end: helper functions ------------------

# 1. Load dataset
dataset= load_my_dataset()
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

# 4. Train and evaluate model
train_dataloader, eval_dataloader = prepare_data_loaders(processed_datasets, tokenizer)
if USING_CROSS_VALIDATION:
    train_and_evaluate_with_KFold()
else:
    train_and_evaluate_model(train_dataloader,eval_dataloader)