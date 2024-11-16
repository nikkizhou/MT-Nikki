import random
import nltk
from nltk.corpus import wordnet

# Download WordNet if not already downloaded
nltk.download('wordnet')

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

def augment_question(question, augmentations=["synonym", "shuffle"], num_synonyms=1):
    """
    Augments a question using specified techniques.
    """
    augmented = question

    if "synonym" in augmentations:
        augmented = synonym_replacement(augmented, n=num_synonyms)

    if "shuffle" in augmentations:
        augmented = shuffle_words(augmented)

    return augmented


def process_excel_file_with_augmentation():
    df = process_excel_file()
    augmented_data = []

    # Apply augmentation to each question
    for index, row in df.iterrows():
        original_question = row["Question"]
        label = row["labels"]

        # Generate augmented questions
        for _ in range(2):  # Create 2 augmented samples per original question
            augmented_question = augment_question(original_question, augmentations=["synonym", "shuffle"])
            augmented_data.append({"Question": augmented_question, "labels": label})

    # Convert augmented data to a DataFrame and append it to the original
    augmented_df = pd.DataFrame(augmented_data)
    df = pd.concat([df, augmented_df], ignore_index=True)

    return df


def load_my_dataset_with_augmentation():
    df = process_excel_file_with_augmentation()
    df['labels'] = df['labels'].astype(int)

    print("DataFrame columns:", df.columns.tolist())

    # Save to a temporary CSV and load as Hugging Face dataset
    csv_file = './data/temp_dataset_augmented.csv'
    df.to_csv(csv_file, index=False)
    dataset = load_dataset('csv', data_files=csv_file)
    return dataset


dataset = load_my_dataset_with_augmentation()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
processed_datasets = tokenize_and_process_dataset(dataset, tokenizer)

train_dataloader, eval_dataloader = prepare_data_loaders(processed_datasets, tokenizer, batch_size=16)
