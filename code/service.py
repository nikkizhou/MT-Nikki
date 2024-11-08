
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datasets import load_dataset
from transformers import DataCollatorWithPadding,AutoTokenizer, AutoModelForSequenceClassification


COMBINE_CATEGORIES = True
USING_CROSS_VALIDATION = True
# MODEL_NAME = 'bert-base-uncased'
# MODEL_NAME = 'meta-llama/Llama-3.2-1B'
MODEL_NAME ='distilbert-base-uncased'


# 1. Get and reorgnize dataframe
excel_file = './data/Categorized_mocks.xlsx'
original_label_columns = ['R2-1', 'R2_2B', 'R2_2D', 'R2_2SD', 'R2_3', 'R2_3YN', 'R2_OP', 
          'R2_4QG', 'R2_4QL', 'R2_4QP', 'R2_4QR', 'R2_4QI', 'R2_4QV', 
          'R2_5', 'R2_6']

combined_label_columns = ['invitation', 'directive', 'option-posing', 'suggestive']

label_columns = combined_label_columns if COMBINE_CATEGORIES else original_label_columns

def process_excel_file():
    df = pd.read_excel(excel_file, header=1)
    df = df.iloc[:, :18]
    df = df.drop(df.index[0])  # Remove the Open-Closed row

    df[original_label_columns] = df[original_label_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    df = df[df[original_label_columns].sum(axis=1) > 0]  # Keep only rows with at least one label > 0

    if COMBINE_CATEGORIES:
        df = combine_categories(df)
    else:
        df['labels'] = df.apply(find_first_label, axis=1)

    # Keep only the relevant columns
    df = df[['Question', 'labels']]
    df = df[df['labels'] >= 0] # Remove rows with no classification

    return df

def find_first_label(row):
    for col in original_label_columns:
        if row[col] == 1:
            return original_label_columns.index(col)
    return None  # Return None if no 1 is found 


def load_my_dataset():
    df = process_excel_file()
    df['labels'] = df['labels'].astype(int)

    print("DataFrame columns:", df.columns.tolist())

    # 2. load dataset
    csv_file = './data/temp_dataset.csv'
    df.to_csv(csv_file, index=False)
    dataset = load_dataset('csv', data_files=csv_file)
    return dataset

def combine_categories(df):
    df['invitation'] = df[['R2-1']].sum(axis=1)
    df['directive'] = df[['R2_2B', 'R2_2D', 'R2_2SD']].sum(axis=1)
    df['option-posing'] = df[['R2_3', 'R2_3YN', 'R2_OP']].sum(axis=1)
    df['suggestive'] = df[['R2_4QG', 'R2_4QL', 'R2_4QP', 'R2_4QR', 'R2_4QI', 'R2_4QV']].sum(axis=1)

    # Drop original columns to avoid confusion
    combined_columns = ['invitation', 'directive', 'option-posing', 'suggestive']
    df = df[combined_columns + ['Question']]  # Keep only the combined columns and the Question column
   
    # Update labels based on combined categories
    df = df.copy()
    df['labels'] = -1  # Initialize labels with -1 to indicate no match
    for idx, category in enumerate(combined_columns):
        df.loc[df[category] > 0, 'labels'] = idx
        df.loc[df['labels'] >= 0, 'labels'] = df['labels'] # Once the first match is found, don't change it (break the loop)

    return df


def print_no_label_samples(df):
    # Identify samples without any label
    no_label_samples = df[df['label'].isnull()]

    # Print samples without any label
    print("Samples without any label:")
    print(no_label_samples)

    # Count of samples without any label
    no_label_count = no_label_samples.shape[0]
    print("Total number of samples without any label:", no_label_count)



def tokenize_and_process_dataset(dataset,tokenizer):
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
