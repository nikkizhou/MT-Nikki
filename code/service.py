
import torch
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


COMBINE_CATEGORIES = True
USING_CROSS_VALIDATION = True
MODEL_NAME = 'bert-base-uncased'
# MODEL_NAME = 'meta-llama/Llama-3.2-1B'
#MODEL_NAME ='distilbert-base-uncased'
DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Get and reorgnize dataframe
file_name1 = './MT/data/Categorized_mocks.xlsx'
file_name2 = "Question Type examples 9_20_24.xlsx"
file_name3 = 'Forensic Trafficking Interviews Question Type Examples 10_1_24.xlsx'

original_label_columns = ['R2-1', 'R2_2B', 'R2_2D', 'R2_2SD', 'R2_3', 'R2_3YN', 'R2_OP', 
          'R2_4QG', 'R2_4QL', 'R2_4QP', 'R2_4QR', 'R2_4QI', 'R2_4QV', 
          'R2_5', 'R2_6']

combined_label_columns = ['open-ended', 'option-posing', 'none-questions', 'leading']

label_columns = combined_label_columns if COMBINE_CATEGORIES else original_label_columns

def read_file (file_name):
    df = pd.read_excel('./MT/data/'+file_name, skiprows=1)  # Skip the first row

    filtered_df = df[['Question', 'Label']]
    filtered_df = filtered_df.dropna(subset=['Label']).reset_index(drop=True)
    filtered_df['Question'] = filtered_df['Question'].str.replace(r'^Q[.:]\s*', '', regex=True)

    # filtered_df.to_csv('./MT/data/csv_'+file_name, index=False)
    # dataset = load_dataset('csv', data_files=filtered_df)
    return filtered_df


# Only for file 1
def combine_categories(df):
    # df['invitation'] = df[['R2-1']].sum(axis=1)
    # df['directive'] = df[['R2_2B', 'R2_2D', 'R2_2SD']].sum(axis=1)
    # df['option-posing'] = df[['R2_3', 'R2_3YN', 'R2_OP']].sum(axis=1)
    # df['suggestive'] = df[['R2_4QG', 'R2_4QL', 'R2_4QP', 'R2_4QR', 'R2_4QI', 'R2_4QV']].sum(axis=1)
    # df['none-questions'] = df[['R2_5']].sum(axis=1)
    # df['multiple']= df[['R2_6']].sum(axis=1)

    df['open-ended'] = df[['R2-1','R2_2B', 'R2_2D', 'R2_2SD']].sum(axis=1)
    df['option-posing'] = df[['R2_3', 'R2_3YN', 'R2_OP']].sum(axis=1)
    df['none-questions'] = df[['R2_5']].sum(axis=1)
    df['suggestive'] = df[['R2_4QG', 'R2_4QL', 'R2_4QP', 'R2_4QR', 'R2_4QI', 'R2_4QV']].sum(axis=1)
    df['multiple']= df[['R2_6']].sum(axis=1)

    # combined_columns_file1 = ['invitation', 'directive', 'option-posing', 'suggestive','none-questions','multiple']
    combined_columns_file1 = ['open-ended', 'option-posing','none-questions',  'suggestive', 'multiple']
    df = df[combined_columns_file1 + ['Question']].copy()  # Keep only the combined columns and the Question column
   
    df['Label'] = -1 
    df['Label'] = df['Label'].astype('object')
    for idx, category in enumerate(combined_columns_file1):
        df.loc[(df['Label'] == -1) & (df[category] > 0), 'Label'] = category
    
    label_counts = df['Label'].value_counts()
    # print('File 1 Combined: ')
    # print(label_counts)
  
    # Filter out rows labeled 'suggestive' and 'multiple'
    df = df[~df['Label'].isin(['suggestive', 'multiple'])] 
    return df


# Only for file 2 and file 3
def map_labels(dataset):
    label_mapping = {
        # 'Leading , tag': 'Leading-tag',
        # 'leading (tag)': 'Leading-tag',
        # 'leading (statement)': 'Leading-statement',
        # 'leading, statement question': 'Leading-statement',
        'Leading , tag': 'leading',
        'leading (tag)': 'leading',
        'leading (statement)': 'leading',
        'leading, statement question': 'leading',
        'option-posing': 'option-posing',
        'option posiing': 'option-posing',
        'option posing ': 'option-posing',
        'option-posing ': 'option-posing',
        'DYK/DYR': 'option-posing',
        'DYK': 'option-posing',
        'open-ended': 'open-ended',
        'Open-ended ': 'open-ended',
    }

    # Map labels to the respective indices in combined_columns
    dataset['Label'] = dataset['Label'].map(label_mapping)

    # label_counts = dataset['Label'].value_counts()
    # print('File 2/3: ')
    # print(label_counts)

    return dataset

def merge_datasets(dataset1, dataset2, dataset3):
    dataset2 = map_labels(dataset2)
    dataset3 = map_labels(dataset3)
    merged_dataset = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)
    return merged_dataset


def process_excel_file():
    df_file1 = pd.read_excel(file_name1, header=1)
    df_file1 = df_file1.iloc[:, :18]
    df_file1 = df_file1.drop(df_file1.index[0])  # Remove the Open-Closed row

    df_file1[original_label_columns] = df_file1[original_label_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_file1 = df_file1[df_file1[original_label_columns].sum(axis=1) > 0]  # Keep only rows with at least one label > 0

    if COMBINE_CATEGORIES:
        dataset1 = combine_categories(df_file1)  # Ensure this function returns a DataFrame
        dataset1 = dataset1.to_pandas() if not isinstance(dataset1, pd.DataFrame) else dataset1 
       
        dataset2 = read_file(file_name2)
        dataset3 = read_file(file_name3)
        
        merged_dataset = merge_datasets(dataset1, dataset2, dataset3)
    
        label_to_int = {label: idx for idx, label in enumerate(combined_label_columns)}
        merged_dataset['Label'] = merged_dataset['Label'].map(label_to_int)
        # print('MERGED: ')
        # print(merged_dataset['Label'].value_counts()) 

        df= merged_dataset

    else:
        df_file1['Label'] = df_file1.apply(find_first_label, axis=1)
        df = df_file1
     
    # Keep only the relevant columns
    df = df[['Question', 'Label']]
    df = df[df['Label'] >= 0] # Remove rows with no classification
    df['Label'] = df['Label'].astype(int)

    return df.dropna().reset_index(drop=True)

process_excel_file()

def find_first_label(row):
    for col in original_label_columns:
        if row[col] == 1:
            return original_label_columns.index(col)
    return None  # Return None if no 1 is found 

def load_my_dataset():
    df = process_excel_file()
    df['Label'] = df['Label'].astype(int)

    #print("DataFrame columns:", df.columns.tolist())

    # 2. load dataset
    csv_file = './MT/data/temp_dataset.csv'
    df.to_csv(csv_file, index=False)
    dataset = load_dataset('csv', data_files=csv_file)
    return dataset

def compute_class_weights(train_dataset):
    try:
        labels = train_dataset['Label']  
    except KeyError:
        labels = train_dataset['labels']  
    class_counts = torch.bincount(torch.tensor(labels).to(DEVICE))
    #print(class_counts)
    class_weights = 1.0 / class_counts.float()
    return class_weights


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
    
    # Print example questions and labels
    # for i in range(example_count):
    #     question = dataset[split_name][i]['Question']
    #     label = processed_datasets[split_name][i]['Label']
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
