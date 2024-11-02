#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/nikkizhou/ML/blob/main/MT_Nikki.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


from tqdm.auto import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset

from sklearn.model_selection import KFold 
from sklearn.metrics import classification_report

from transformers import get_scheduler
from transformers import DataCollatorWithPadding,AutoTokenizer, AutoModelForSequenceClassification
from service import COMBINE_CATEGORIES,USING_CROSS_VALIDATION,label_columns,process_excel_file


# --------------- start: helper functions -----------------
def tokenize_and_process_dataset(dataset):
    # Function to tokenize questions, including tokenizer as a parameter
    def tokenize_questions(examples):
        return tokenizer(examples['Question'], padding=True, truncation=True)

    # Tokenize the questions
    tokenize_datasets = dataset.map(tokenize_questions, batched=True)
    processed_datasets = tokenize_datasets.remove_columns(['Question'])
    
    #print("Column names: " + str(processed_datasets[split_name].column_names))

    # Print example questions and labels
    # for i in range(example_count):
    #     question = dataset[split_name][i]['Question']
    #     label = processed_datasets[split_name][i]['labels']
    #     print(f"Question: {question}, Label: {label}, Type: {str(type(label))}")
    return processed_datasets

def prepare_data_loaders(processed_datasets, tokenizer, test_size=0.2, batch_size=4, seed=42):
    # 1. Split the dataset into training and test sets
    datasets = processed_datasets['train'].train_test_split(test_size=test_size, seed=seed)

    # 2. Get data_collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Get train_dataloader and eval_dataloader
    train_dataloader = DataLoader(datasets['train'], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(datasets['test'], batch_size=batch_size, collate_fn=data_collator)
    
    return train_dataloader, eval_dataloader

def train_model(model, train_dataloader, num_epochs, gradient_accumulation_steps, lr=5e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            #outputs = model(**batch)
            outputs = model(**batch) 
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

def evaluate_model(model, eval_dataloader, label_columns):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    print_classification_report(all_labels, all_predictions, label_columns)

    return accuracy

 
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
        all_labels, all_predictions, labels=label_indices, target_names=label_columns
    )
    print(report)


def train_and_evaluate_model(train_dataloader,eval_dataloader):
    #print("Unique labels in dataset:", df['labels'].unique())
    train_model(model, train_dataloader, num_epochs=3, gradient_accumulation_steps=4)
    accuracy  = evaluate_model(model, eval_dataloader, label_columns)
    return accuracy

def prepare_data_loaders_for_kfold(dataset, tokenizer, batch_size=4):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    
    return dataloader

def train_and_evaluate_with_KFold():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    full_dataset = processed_datasets['train']

    # Extract inputs and labels
    inputs = full_dataset.remove_columns('labels')
    labels = full_dataset['labels']

    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(inputs)):
        print(f"\nFold {fold + 1}/{kf.n_splits}")

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
df = process_excel_file()
df['labels'] = df['labels'].astype(int)

print("DataFrame columns:", df.columns.tolist())

# 2. load dataset
csv_file = '../data/temp_dataset.csv'
df.to_csv(csv_file, index=False)
dataset = load_dataset('csv', data_files=csv_file)

# 3. Tokenize and process dataset
tokenizer_name='bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
processed_datasets = tokenize_and_process_dataset(dataset)


# 4. Set Up the Optimizer and Learning Rate Scheduler
model_name = "bert-base-uncased"
num_labels = 4 if COMBINE_CATEGORIES else 15
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 5. Train and evaluate model
if USING_CROSS_VALIDATION:
    train_and_evaluate_with_KFold()
else:
    train_dataloader, eval_dataloader = prepare_data_loaders(processed_datasets, tokenizer)
    train_and_evaluate_model(train_dataloader,eval_dataloader)