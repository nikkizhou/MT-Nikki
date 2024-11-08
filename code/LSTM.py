from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from service import label_columns,tokenize_and_process_dataset,prepare_data_loaders,load_my_dataset
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --------------- start: helper functions -----------------

def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_columns, yticklabels=label_columns)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


# def train_model(model, train_dataloader, num_epochs=3, lr=5e-5):
#     optimizer = optim.AdamW(model.parameters(), lr=lr)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     for epoch in range(num_epochs):
#         model.train()
#         all_labels = []
#         all_preds = []
        
#         for batch in train_dataloader:
#             optimizer.zero_grad()
            
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
            
#             # Forward pass
#             outputs = model(input_ids)
#             loss_fn = nn.CrossEntropyLoss()
#             loss = loss_fn(outputs, labels)
            
#             # Collect labels and predictions
#             _, preds = torch.max(outputs, dim=1)
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())

#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()

#         # Compute and display classification report
#         report = classification_report(all_labels, all_preds, target_names=label_columns, digits=2,zero_division=0)
#         print(f"Classification Report Epoch {epoch}: \n{report}")

#         # Compute and plot confusion matrix
#         cm = confusion_matrix(all_labels, all_preds)
#         plot_confusion_matrix(cm)
def evaluate_model(model, eval_dataloader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids)
            _, preds = torch.max(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=label_columns, digits=2, zero_division=0)
    print(f"Validation Classification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm)

# Integrate within the training loop
def train_and_evaluate_model(model, train_dataloader, eval_dataloader, num_epochs = 3, lr=5e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(3):
        model.train()
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Call the evaluation after each epoch
        evaluate_model(model, eval_dataloader)

# --------------- end: helper functions -----------------

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        hidden = h_n[-1]
        output = self.fc(self.dropout(hidden))
        return output
    

# 1. Load dataset
dataset= load_my_dataset()

# 2. Tokenize and process dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
processed_datasets = tokenize_and_process_dataset(dataset, tokenizer)


# 3. Model Parameters
vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 256
output_dim = 4  # For binary classification (modify as per your dataset)

model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)

# 4. Prepare the data loaders and train
train_dataloader, eval_dataloader = prepare_data_loaders(processed_datasets, tokenizer)
train_and_evaluate_model(model, train_dataloader, eval_dataloader, label_columns)


