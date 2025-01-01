import os
import torch
import pandas as pd
from datasets import load_dataset,Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


COMBINE_CATEGORIES = True
USING_CROSS_VALIDATION = False
# MODEL_NAME = 'bert-base-uncased'
MODEL_NAME = 'meta-llama/Llama-3.2-1B'
# MODEL_NAME ='distilbert-base-uncased'
DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_names_mapping = {
    'bert-base-uncased': 'BERT',
    'meta-llama/Llama-3.2-1B': 'Llama',
    'distilbert-base-uncased': 'DistilBert'
}

model_name_simplified = model_names_mapping.get(MODEL_NAME, 'Unknown Model')


file_name1 = './MT/data/original_data/Categorized_mocks.xlsx'
file_name2 = "Question Type examples 9_20_24.xlsx"
file_name3 = 'Forensic Trafficking Interviews Question Type Examples 10_1_24.xlsx'

synthetic_data_path='./MT/data/synthetic_data/synthetic_GPT_new.csv'
original_synthetic_data_path = './MT/data/original_synthetic_data/combined_dataset.csv'


original_label_columns = ['R2-1', 'R2_2B', 'R2_2D', 'R2_2SD', 'R2_3', 'R2_3YN', 'R2_OP', 
          'R2_4QG', 'R2_4QL', 'R2_4QP', 'R2_4QR', 'R2_4QI', 'R2_4QV', 
          'R2_5', 'R2_6']

combined_label_columns = ['open-ended', 'option-posing', 'none-questions', 'leading']

label_columns = combined_label_columns if COMBINE_CATEGORIES else original_label_columns

# For file 2 and file 3
def read_file (file_name):
    df = pd.read_excel('./MT/data/original_data/'+file_name, skiprows=1)  # Skip the first row

    filtered_df = df[['Question', 'Label']]
    filtered_df = filtered_df.dropna(subset=['Label']).reset_index(drop=True)
    filtered_df['Question'] = filtered_df['Question'].str.replace(r'^Q[.:]\s*', '', regex=True)

    # filtered_df.to_csv('./MT/data/original_data/csv_'+file_name, index=False)
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

# Merge data from file 1, file 2 and file 3
def merge_datasets(df_file1):
    dataset1 = combine_categories(df_file1)  # Ensure this function returns a DataFrame
    dataset1 = dataset1.to_pandas() if not isinstance(dataset1, pd.DataFrame) else dataset1 
    
    dataset2 = read_file(file_name2)
    dataset3 = read_file(file_name3)

    dataset2 = map_labels(dataset2)
    dataset3 = map_labels(dataset3)

    merged_dataset = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)

    label_to_int = {label: idx for idx, label in enumerate(combined_label_columns)}
    merged_dataset['Label'] = merged_dataset['Label'].map(label_to_int)
      
    return merged_dataset


# def add_synthetic_data(merged_dataset):
#     synthetic_df = pd.read_csv(synthetic_data_path)
#     combined_df = pd.concat([merged_dataset, synthetic_df], ignore_index=True)
#     return combined_df

def add_synthetic_data_to_train_set(merged_dataset, synthetic_df):
    
    train_df, test_df = train_test_split(merged_dataset, test_size=0.2, random_state=42)
    combined_train_df = pd.concat([train_df, synthetic_df], ignore_index=True)
    
    # Verify that all syntheic data are in training set.
    train_synthetic_count = combined_train_df[combined_train_df.index.isin(synthetic_df.index)].shape[0]
    print(f"Total synthetic samples: {len(synthetic_df)}")
    print(f"Synthetic samples in training set: {train_synthetic_count}")
   
    return combined_train_df, test_df

def preprocess_dataframe(df):
    # Keep only 'Question' and 'Label' columns and remove invalid labels
    df = df[['Question', 'Label']]
    df = df[df['Label'] >= 0]
    df['Label'] = df['Label'].astype(int)
    return df.dropna().reset_index(drop=True)

def get_test_and_train_df():
    df_file1 = pd.read_excel(file_name1, header=1)
    df_file1 = df_file1.iloc[:, :18]
    df_file1 = df_file1.drop(df_file1.index[0])  # Remove the Open-Closed row

    df_file1[original_label_columns] = df_file1[original_label_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_file1 = df_file1[df_file1[original_label_columns].sum(axis=1) > 0]  # Keep only rows with at least one label > 0

    # if COMBINE_CATEGORIES:
    merged_df = merge_datasets(df_file1)
    synthetic_df = pd.read_csv(synthetic_data_path)

    train_df, test_df = add_synthetic_data_to_train_set(merged_df, synthetic_df)
     
    # Keep only the relevant columns
    train_df = preprocess_dataframe(train_df)
    test_df = preprocess_dataframe(test_df)

    print('Label Count Training Set: ')
    print(train_df['Label'].value_counts()) 
    print('Label Count Test Set: ')
    print(test_df['Label'].value_counts())

    # train_df.to_csv(original_synthetic_data_path, index=False)

    return train_df, test_df

def add_quotes(question):
    if not (question.startswith('"') and question.endswith('"')):
        return f'"{question}"'
    return question
    
    
# def test():
#     df = pd.read_csv('./MT/data/synthetic_data/synthetic_GPT.csv')

 
#     # Identify the indices of rows to delete
#     label_2_indices = df[(df['Label'] == 2)].index[::4]
#     if len(label_2_indices) > 435:
#         label_2_indices = label_2_indices[:435]

#     label_0_indices = df[(df['Label'] == 0)].index[::3]
#     if len(label_0_indices) > 140:
#         label_0_indices = label_0_indices[:140]

#     label_3_indices = df[(df['Label'] == 3)].index[::3]
#     if len(label_3_indices) > 18:
#         label_3_indices = label_3_indices[:18]

#     df.drop(label_0_indices, inplace=True)
#     df.drop(label_2_indices, inplace=True)
#     df.drop(label_3_indices, inplace=True)

#     # Save the updated DataFrame to a new CSV file
#     df.to_csv('./MT/data/synthetic_data/synthetic_GPT_new.csv', index=False)

#     # Display the updated DataFrame
#     print(df.head())

# test()

def find_first_label(row):
    for col in original_label_columns:
        if row[col] == 1:
            return original_label_columns.index(col)
    return None  # Return None if no 1 is found 


# Load dataset for all 3 files combined with the synthetic data
def load_my_dataset():
    train_df,test_df  = get_test_and_train_df()
    test_dataset = Dataset.from_pandas(test_df)
    train_dataset = Dataset.from_pandas(train_df)
    return DatasetDict({'train': train_dataset, 'test': test_dataset})


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
    return processed_datasets

def prepare_data_loaders(processed_datasets, tokenizer, test_size=0.2, batch_size=4, seed=42):
    # 1. Split the dataset into training and test sets
    #datasets = processed_datasets['train'].train_test_split(test_size=test_size, seed=seed)

    # 2. Get data_collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Get train_dataloader and eval_dataloader
    train_dataloader = DataLoader(processed_datasets['train'], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(processed_datasets['test'], batch_size=batch_size, collate_fn=data_collator)

    return train_dataloader, eval_dataloader

def plot_confusion_matrix(all_labels, all_predictions, label_columns, output_filename, title):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_columns)
    plt.figure(figsize=(8, 8))
    disp.plot(cmap='viridis', xticks_rotation='vertical')
    plt.title(title)
    plt.tight_layout()

    current_path = os.getcwd()
    output_path = os.path.join(current_path, output_filename)

    plt.savefig(output_path)  
    print(f"Confusion matrix saved to {output_path}")
    plt.close()  # Close the plot to free resources