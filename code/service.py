import os
import torch
import pandas as pd
from datasets import load_dataset,Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold

COMBINE_CATEGORIES = True
USING_CROSS_VALIDATION = True
ADD_SYNTHETIC_DATA = False

# MODEL_NAME ='distilbert-base-uncased'
# MODEL_NAME = 'bert-base-uncased'
MODEL_NAME = 'meta-llama/Llama-3.2-1B'
model_names_mapping = {
    'bert-base-uncased': 'BERT',
    'meta-llama/Llama-3.2-1B': 'Llama',
    'distilbert-base-uncased': 'DistilBert'
}
model_name_simplified = model_names_mapping.get(MODEL_NAME, 'Unknown Model')

DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_name1 = './MT/data/original_data/Categorized_mocks.xlsx'
file_name2 = "Question Type examples 9_20_24.xlsx"
file_name3 = 'Forensic Trafficking Interviews Question Type Examples 10_1_24.xlsx'

original_label_columns = ['R2-1', 'R2_2B', 'R2_2D', 'R2_2SD', 'R2_3', 'R2_3YN', 'R2_OP', 
          'R2_4QG', 'R2_4QL', 'R2_4QP', 'R2_4QR', 'R2_4QI', 'R2_4QV', 
          'R2_5', 'R2_6']
combined_label_columns = ['open-ended', 'option-posing', 'none-questions', 'leading']
label_columns = combined_label_columns if COMBINE_CATEGORIES else original_label_columns

synthetic_data_path='./MT/data/synthetic_data/synthetic_GPT_new.csv'
original_synthetic_data_path = './MT/data/original_synthetic_data/combined_dataset.csv'


def get_fold_string(fold):
    return f"Fold {fold + 1}" if fold is not None else ""

# For file 2 and file 3
def read_file (file_name):
    df = pd.read_excel('./MT/data/original_data/'+file_name, skiprows=1)  # Skip the first row

    filtered_df = df[['Question', 'Label']]
    filtered_df = filtered_df.dropna(subset=['Label']).reset_index(drop=True)
    filtered_df['Question'] = filtered_df['Question'].str.replace(r'^Q[.:]\s*', '', regex=True)

    # filtered_df.to_csv('./MT/data/original_data/csv_'+file_name, index=False)
    # dataset = load_dataset('csv', data_files=filtered_df)
    return filtered_df

def get_df_file1():
    df_file1 = pd.read_excel(file_name1, header=1)
    df_file1 = df_file1.iloc[:, :18]
    df_file1 = df_file1.drop(df_file1.index[0])  # Remove the Open-Closed row

    df_file1[original_label_columns] = df_file1[original_label_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_file1 = df_file1[df_file1[original_label_columns].sum(axis=1) > 0]  # Keep only rows with at least one label > 0
    
    return df_file1


# Only for file 1
def combine_categories(df):
    # df['invitation'] = df[['R2-1']].sum(axis=1)
    # df['directive'] = df[['R2_2B', 'R2_2D', 'R2_2SD']].sum(axis=1)
    # df['option-posing'] = df[['R2_3', 'R2_3YN', 'R2_OP']].sum(axis=1)
    # df['suggestive'] = df[['R2_4QG', 'R2_4QL', 'R2_4QP', 'R2_4QR', 'R2_4QI', 'R2_4QV']].sum(axis=1)
    # df['none-questions'] = df[['R2_5']].sum(axis=1)
    # df['multiple']= df[['R2_6']].sum(axis=1)
    # columns_to_sum = ['R2-1', 'R2_2B', 'R2_2D', 'R2_2SD']
    # df[columns_to_sum] = df[columns_to_sum].apply(pd.to_numeric, errors='coerce').fillna(0)

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
def merge_datasets():
    df_file1 = get_df_file1()
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
    df = df[['Question', 'Label', 'is_synthetic']] if 'is_synthetic' in df.columns and USING_CROSS_VALIDATION and ADD_SYNTHETIC_DATA else df[['Question', 'Label']]
    df = df[df['Label'] >= 0]
    df['Label'] = df['Label'].astype(int)
    return df.dropna().reset_index(drop=True)

def get_test_and_train_df():
    # if COMBINE_CATEGORIES:
    merged_df = merge_datasets()

    if ADD_SYNTHETIC_DATA:
        synthetic_df = pd.read_csv(synthetic_data_path)
        train_df, test_df = add_synthetic_data_to_train_set(merged_df, synthetic_df)
    else:
        train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)
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
    

def find_first_label(row):
    for col in original_label_columns:
        if row[col] == 1:
            return original_label_columns.index(col)
    return None  # Return None if no 1 is found 


# Load dataset for all 3 files combined with the synthetic data
def load_and_split_dataset():
    train_df,test_df  = get_test_and_train_df()
    test_dataset = Dataset.from_pandas(test_df)
    train_dataset = Dataset.from_pandas(train_df)
    return DatasetDict({'train': train_dataset, 'test': test_dataset})

def load_and_mark_potential_synthetic_data():
    real_world_df = merge_datasets()
    real_world_df = preprocess_dataframe(real_world_df)
    real_world_dataset = Dataset.from_pandas(real_world_df)

    if not ADD_SYNTHETIC_DATA: return real_world_dataset

    synthetic_df = pd.read_csv(synthetic_data_path)
    print(f"Total synthetic samples: {len(synthetic_df)}")
    synthetic_df['is_synthetic'] = True
    real_world_df['is_synthetic'] = False

    combined_real_and_synthetic_df = pd.concat([real_world_df, synthetic_df], ignore_index=True)
    combined_real_and_synthetic_df = preprocess_dataframe(combined_real_and_synthetic_df)
    combined_real_and_synthetic_dataset = Dataset.from_pandas(combined_real_and_synthetic_df)

    return  combined_real_and_synthetic_dataset


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


def prepare_data_loaders_for_kfold(dataset, tokenizer, batch_size=4):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    
    return dataloader

def get_inputs_and_labels(full_dataset):
    label_column = 'labels' if 'labels' in full_dataset.column_names else 'Label'
    synthetic_column = 'is_synthetic' if ADD_SYNTHETIC_DATA else None
    
    columns_to_remove = [label_column]
    if synthetic_column:
        columns_to_remove.append(synthetic_column)
    
    inputs = full_dataset.remove_columns(columns_to_remove)
    labels = full_dataset[label_column]
    
    return inputs, labels

def train_and_evaluate_with_KFold(full_dataset, train_and_evaluate_model, tokenizer):
    #kf = KFold(n_splits=3, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Extract inputs and labels
    inputs, labels = get_inputs_and_labels(full_dataset)
    
    # Identify indices for real-world data
    real_world_indices = (
    {i for i, example in enumerate(full_dataset) if not example['is_synthetic']}
    if ADD_SYNTHETIC_DATA
    else set(range(len(full_dataset)))
)

    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(inputs, labels)):
        # Filter validation indices to include only real-world data
        # Which makes validation set  smaller
        val_idx = [idx for idx in val_idx if idx in real_world_indices]
        print(f"Fold {fold + 1}/{skf.n_splits} - Training set size: {len(train_idx)}, Validation set size: {len(val_idx)}")
        
        # Create train and validation datasets
        # training set would contain both real world and synthetic data
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
        eval_dataloader_kFold = prepare_data_loaders_for_kfold(val_dataset, tokenizer)

        accuracy = train_and_evaluate_model(train_dataloader_kFold, eval_dataloader_kFold, fold,train_dataset)
        accuracies.append(accuracy)

    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"\nAverage Accuracy across all folds: {average_accuracy:.4f}")