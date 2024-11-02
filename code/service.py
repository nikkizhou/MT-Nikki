
import pandas as pd


COMBINE_CATEGORIES = True
USING_CROSS_VALIDATION = True
# 1. Get and reorgnize dataframe
excel_file = '../data/Categorized_mocks.xlsx'
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
