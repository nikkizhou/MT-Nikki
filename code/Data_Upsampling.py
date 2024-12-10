import nltk
import random
import os
import pandas as pd
from nltk.corpus import wordnet
from datasets import load_dataset
from service import MODEL_NAME, combined_label_columns,synthetic_data_path
from transformers import AutoTokenizer
import openai
from openai import OpenAI
from dotenv import load_dotenv
from datasets import Dataset
import math

load_dotenv()

# Download WordNet if not already downloaded
# nltk.download('wordnet')
# openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI( api_key= os.getenv('OPENAI_API_KEY'))


def getPrompt(question, label):

    return f"""
    You are tasked with generating synthetic questions for a dataset used in a question classification task. 
    The questions are taken from interviews conducted between child protection personnel and mistreated children.
    Below are the instructions you need to follow:

    1. You will be provided with an input question and its corresponding question type. The types are:
    - Question Type Index 0, open-ended: Questions that encourage elaborate or unrestricted responses. For example: Can you tell me everything that happened?
    - Question Type Index 1, option-posing: Questions that present specific options or choices. For example: Are you standing or sitting?
    - Question Type Index 2, leading: Questions that suggest or imply a particular answer. For example:  DID SOMETHING BAD HAPPEN TO YOU WHEN YOU WERE WITH YOUR BROTHER?
    - Question Type Index 3, none-questions: Statements or expressions that do not fit into the above categories.

    2. Your task is to rewrite the question while preserving its original type and intent. Ensure the following:  
    - The structure of the question matches the specified question type.  
    - The content remains relevant to the context of interviews with mistreated children.  
    - The phrasing is sufficiently altered to qualify as a new, synthetic question while retaining the original meaning.  

    3. Avoid simply replacing words with synonyms. Instead, aim to reframe or rephrase the question naturally while keeping its meaning and type intact.

    Here I'm providing the Input Question and it's corresponding Question Type Index:
    Input Question: "{question}"
    Question Type Index: "{label}"

    Please provide the generated question based on the provided info above directly without any extra word.
    """


def synonym_replacement(sentence, n=1):
    """
    Replaces n words in the sentence with their synonyms.
    """
    words = sentence.split()
    new_words = words.copy()
    random.shuffle(words)
    num_replaced = 0

    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if w == word and num_replaced < n else w for w in new_words]
            num_replaced += 1

        if num_replaced >= n:
            break

    return " ".join(new_words)

def shuffle_words(sentence):
    """
    Randomly shuffles the words in a sentence while keeping the structure similar.
    """
    words = sentence.split()
    random.shuffle(words)
    return " ".join(words)


# Limits: https://platform.openai.com/settings/organization/limits
def generate_synthetic_question_gpt(question, label):
    """
    Generates synthetic questions using GPT.
    """
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": getPrompt(question,label),
            }
        ],
        model="gpt-4o-mini",
    )
    with open("output.txt", "a") as file:
        file.write(f"Q: {response.choices[0].message.content.strip()} L: {label}\n")
   
    return  response.choices[0].message.content.strip()
    

def augment_question(question,label, augmentations=["synonym", "shuffle", "gpt"], num_synonyms=1):
    """
    Augments a question using specified techniques including GPT.
    """
    synthetic_questions = []

    # if "synonym" in augmentations:
    #     synthetic_questions.append(synonym_replacement(question, n=num_synonyms))

    # if "shuffle" in augmentations:
    #     synthetic_questions.append(shuffle_words(question))

    if "gpt" in augmentations:
        synthetic_questions.append(generate_synthetic_question_gpt(question, label))

    return synthetic_questions


def augment_each_question_with_synthetic_data(df):
    synthetic_data = []
    # Apply augmentation to each question
    for index, row in df.iterrows():
        #if index>3837:
            original_question = row["Question"]
            label = row["Label"]

            # Generate synthetic questions
            synthetic_question = augment_question(original_question, label, augmentations=["synonym", "shuffle","gpt"])
            synthetic_data.append({"Question": synthetic_question, "Label": label})
    
    return synthetic_data

# def upsample_to_balanced_data(df):
#     #upsampled_frames = []
#     max_size = df['Label'].value_counts().max()  # Determine the target size (equal to the largest category size)

#     for category in combined_label_columns:
#         if category == 'open-ended':
#             continue

#         category_df = df[df['Label'] == combined_label_columns.index(category)]
#         current_size = len(category_df)
#         additional_samples_needed_for_each_question = math.ceil((max_size - current_size)/current_size)
        
#         synthetic_data = []
#         for _ in range(additional_samples_needed_for_each_question):
#             # Randomly sample a question from the current category
#             original_row = category_df.sample(1, random_state=42).iloc[0]
#             question = original_row['Question']
#             label = original_row['Label']
            
#             # Generate a synthetic question
#             synthetic_question = generate_synthetic_question_gpt(question, label)
#             synthetic_data.append({"Question": synthetic_question, "Label": label})     

def upsample_to_balanced_data(df): 
    max_size = df['Label'].value_counts().max()  # Determine the target size (equal to the largest category size)

    for category in combined_label_columns:
        if category == 'open-ended':
            continue

        # Filter rows for the current category
        category_df = df[df['Label'] == combined_label_columns.index(category)]
        current_size = len(category_df)
        additional_samples_needed_for_each_question = math.ceil((max_size - current_size) / current_size)
        
        synthetic_data = []
        for _, row in category_df.iterrows():  # Iterate through every row in the current category
            question = row['Question']
            label = row['Label']
            
            for _ in range(additional_samples_needed_for_each_question):
                # Generate a synthetic question
                synthetic_question = generate_synthetic_question_gpt(question, label)
                synthetic_data.append({"Question": synthetic_question, "Label": label})
        
        # Optionally append the synthetic data to the dataframe
        #df = df.append(pd.DataFrame(synthetic_data), ignore_index=True)
        
    return df

def process_excel_file_with_augmentation():
    df = get_test_and_train_df()
    # synthetic_data = augment_each_question_with_synthetic_data(df)
    synthetic_data = upsample_to_balanced_data(df)

    # Convert synthetic data to a DataFrame and append it to the original
    synthetic_df = pd.DataFrame(synthetic_data)
    df = pd.concat([df, synthetic_df], ignore_index=True)
    
    return df


def load_my_dataset_with_augmentation():
    df = process_excel_file_with_augmentation()
    df['Label'] = df['Label'].astype(int)
    
    print("DataFrame columns:", df.columns.tolist())

    # Save to a temporary CSV and load as Hugging Face dataset
    df.to_csv(synthetic_data_path, index=False)
    dataset = load_dataset('csv', data_files=synthetic_data_path)
    return dataset


dataset = load_my_dataset_with_augmentation()
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# processed_datasets = tokenize_and_process_dataset(dataset, tokenizer)

# train_dataloader, eval_dataloader = prepare_data_loaders(processed_datasets, tokenizer, batch_size=16)

