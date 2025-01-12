from transformers import AutoTokenizer
from service import DEVICE, USING_CROSS_VALIDATION,label_columns,plot_confusion_matrix,tokenize_and_process_dataset,prepare_data_loaders,load_and_split_dataset,compute_class_weights,train_and_evaluate_with_KFold,load_and_mark_potential_synthetic_data,get_fold_string
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim


# --------------- start: helper functions -----------------

def getLabels(batch):
    try:
        labels = batch['labels'].to(DEVICE)
    except KeyError:
        labels = batch['Label'].to(DEVICE)
    return labels

   
def evaluate_model(model, eval_dataloader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = getLabels(batch)
            outputs = model(input_ids=input_ids)
            predictions = torch.argmax(outputs, dim=-1)
            
            # Update metrics
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    
    return accuracy, all_predictions, all_labels

def train_and_evaluate_model( train_dataloader, eval_dataloader,fold=None, train_dataset=None, num_epochs=3, lr=2e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.to(DEVICE)

    class_weights = compute_class_weights(train_dataset).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # for epoch in range(num_epochs):
    model.train()
    
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        labels = getLabels(batch)
        
        # Forward pass
        outputs = model(input_ids)
        loss = loss_fn(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Evaluate model performance after each epoch
    (accuracy,all_predictions,all_labels)  = evaluate_model(model, eval_dataloader)
    print(f"Accuracy {get_fold_string(fold)}: {accuracy:.4f}")
   
#     # Classification report
    report = classification_report(all_labels, all_predictions, target_names=label_columns, digits=2, zero_division=0)
    print(f"Validation Classification Report {get_fold_string(fold)}:\n{report}")

    # Confusion matrix
    output_file = f'CM_BiLSTM_{get_fold_string(fold)}.png' if USING_CROSS_VALIDATION else 'CM_BiLSTM.png'
    title = f'Confusion Matrix BiLSTM {get_fold_string(fold)}' if USING_CROSS_VALIDATION else 'Confusion Matrix BiLSTM'  
    plot_confusion_matrix(all_labels, all_predictions, label_columns,output_file, title)

    return accuracy


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

dataset=  load_and_mark_potential_synthetic_data() if USING_CROSS_VALIDATION else load_and_split_dataset() 
dataset = dataset.rename_column('Label', 'labels')


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

if USING_CROSS_VALIDATION:
    train_and_evaluate_with_KFold(processed_datasets,train_and_evaluate_model,tokenizer)
else:
    train_dataloader, eval_dataloader = prepare_data_loaders(processed_datasets, tokenizer)
    train_and_evaluate_model(train_dataloader,eval_dataloader, None, processed_datasets['train'])