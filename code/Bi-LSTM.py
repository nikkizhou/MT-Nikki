from transformers import AutoTokenizer
from service_copy import DEVICE,label_columns,tokenize_and_process_dataset,prepare_data_loaders,load_my_dataset,compute_class_weights
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim


# --------------- start: helper functions -----------------

def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_columns, yticklabels=label_columns)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def evaluate_model(model, eval_dataloader):
    model.eval()
    
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['Label'].to(DEVICE)
            
            # Forward pass
            outputs = model(input_ids)
            _, preds = torch.max(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=label_columns, digits=2, zero_division=0)
    print(f"Validation Classification Report Epoch:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm)

def train_and_evaluate_model(model, train_dataloader, eval_dataloader, num_epochs=5, lr=2e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.to(DEVICE)
    
    # Loss function with class weights
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(num_epochs):
        model.train()
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['Label'].to(DEVICE)
            
            # Forward pass
            outputs = model(input_ids)
            loss = loss_fn(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Evaluate model performance after each epoch
        evaluate_model(model, eval_dataloader)

#  --------------- end: helper functions -----------------


# LSTM model with bidirectional option
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # hidden_dim * 2 for bidirectional
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        output = self.fc(self.dropout(hidden))
        return output


# 1. Load dataset
dataset= load_my_dataset()

# 3. Compute class weights: Address Class Imbalance with Weighted Loss
class_weights = compute_class_weights(dataset['train'])
#class_weights = compute_class_weights(processed_datasets['train']).to(DEVICE)

# 2. Tokenize and process dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
processed_datasets = tokenize_and_process_dataset(dataset, tokenizer)


# 3. Model Parameters
vocab_size = tokenizer.vocab_size
embedding_dim = 256
hidden_dim = 512
output_dim = 4 

# 4. Train and evaluate the fine-tuned model with class weights
model = BiLSTMModel(vocab_size, embedding_dim=256, hidden_dim=512, output_dim=4)
train_dataloader, eval_dataloader = prepare_data_loaders(processed_datasets, tokenizer)
train_and_evaluate_model(model, train_dataloader, eval_dataloader)