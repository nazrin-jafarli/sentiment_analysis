import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from kmeans import KMeansSemiSupervised  # Import your KMeansSemiSupervised class
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dist_measure='euclidean'  # euclidean or cosine 


# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            # pad_to_max_length=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # 'label': torch.tensor(label, dtype=torch.long)
            'label': label.clone().detach().long()

        }


def get_embeddings(texts, model, device):
    embeddings = []
    for text in texts:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

        # Move inputs to the specified device
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the output embeddings from the hidden states
        hidden_states = outputs.hidden_states
        cls_embedding = hidden_states[-1][:, 0, :].squeeze().cpu().numpy()  # Extract from the last layer's hidden states

        embeddings.append(cls_embedding)
    
    return np.array(embeddings)


# Function to load data from text files in a folder
def load_data_from_folder(folder_path, label):
    texts = []
    labels = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            text = file.read().lower()
            texts.append(text)
            labels.append(label)
    return texts, labels


def evaluate_model(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    test_total = 0
    test_correct = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch_data = batch['input_ids'].to(device)
            batch_attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)

            batch_outputs = model(input_ids=batch_data, attention_mask=batch_attention_mask)
            batch_loss = criterion(batch_outputs.logits, batch_labels)
            test_loss += batch_loss.item()
            _, predicted = torch.max(batch_outputs.logits, 1)
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(batch_labels.cpu().numpy())
    test_loss /= len(test_loader)
    test_accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    return test_loss, test_accuracy, precision, recall, f1




def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            batch_data = batch['input_ids'].to(device)
            batch_attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)

            optimizer.zero_grad()
            batch_outputs = model(input_ids=batch_data, attention_mask=batch_attention_mask)
            batch_loss = criterion(batch_outputs.logits, batch_labels)
            batch_loss.backward()
            optimizer.step()
            epoch_train_loss += batch_loss.item()
        train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        val_total = 0
        val_correct = 0
        val_labels_list = []  # List to store validation labels
        predicted_list = []  
        with torch.no_grad():
            for batch in val_loader:
                batch_data = batch['input_ids'].to(device)
                batch_attention_mask = batch['attention_mask'].to(device)
                batch_labels = batch['label'].to(device)

                batch_outputs = model(input_ids=batch_data, attention_mask=batch_attention_mask)
                batch_loss = criterion(batch_outputs.logits, batch_labels)
                epoch_val_loss += batch_loss.item()

                _, predicted = torch.max(batch_outputs.logits, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
                val_labels_list.extend(batch_labels.cpu().numpy())  # Append batch labels to the list
                predicted_list.extend(predicted.cpu().numpy()) 
        
            
            val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(val_loss)
            val_accuracy = accuracy_score(val_labels_list, predicted_list)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

    
    return train_losses, val_losses



if __name__ == '__main__':

    # Define training parameters
    MAX_LEN = 256
    BATCH_SIZE = 8
    LR = 3e-5 # 2e-5 or 3e-5
    EPOCHS = 15 # 10 or 15

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, output_hidden_states=True)  # 3 classes: positive, negative, neutral
    # model.to(device)


    # Define model types and corresponding tokenizers
    model_types = {
        'BERT': (BertForSequenceClassification, BertTokenizer, 'bert-base-uncased'),
        'RoBERTa': (RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base'),
        'DistilBERT': (DistilBertForSequenceClassification, DistilBertTokenizer, 'distilbert-base-uncased')
    }

    chosen_model_type = 'BERT'  # Change this to 'BERT' or 'RoBERTa' or 'DistilBERT' as needed

    # Initialize tokenizer and model based on chosen model type
    model_class, tokenizer_class, pretrained_weights = model_types[chosen_model_type]
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights, num_labels=3, output_hidden_states=True)
    model.to(device)


    # Load data from positive, negative, and neutral folders
    negative_texts, negative_labels = load_data_from_folder('./main_eng_data/negative', label=0)
    neutral_texts, neutral_labels = load_data_from_folder('./main_eng_data/neutral', label=1)
    positive_texts, positive_labels = load_data_from_folder('./main_eng_data/positive', label=2)

    # Load unlabeled data (replace this with your method of loading unlabeled data)
    unlabeled_texts, _ = load_data_from_folder('./main_eng_data/unlabelled', label=None)

    positive_embeddings = get_embeddings(positive_texts, model, device)
    negative_embeddings = get_embeddings(negative_texts, model, device)
    neutral_embeddings = get_embeddings(neutral_texts, model, device)
    
    # labeled_embeddings = torch.cat([negative_embeddings,neutral_embeddings,positive_embeddings])
    labeled_embeddings = torch.cat([torch.tensor(negative_embeddings), torch.tensor(neutral_embeddings), torch.tensor(positive_embeddings)])

    labels = torch.cat([torch.zeros(len(negative_embeddings)),torch.ones(len(neutral_embeddings)), torch.full((len(positive_embeddings),), 2)])

    unlabeled_embeddings = get_embeddings(unlabeled_texts, model, device)
    unlabeled_embeddings=torch.tensor(unlabeled_embeddings)

    
    clustering_model = KMeansSemiSupervised(n_clusters=3, max_iter=100, distance_measure=dist_measure)

    clustering_model.fit(labeled_embeddings, unlabeled_embeddings, labels)

    # Get pseudo-labels for unlabeled data
    predicted_labels = clustering_model.predict(unlabeled_embeddings)
    predicted_labels=torch.tensor(predicted_labels)
    num_clusters = clustering_model.n_clusters

    all_embeddings = torch.cat([labeled_embeddings, unlabeled_embeddings])
    
    all_labels = torch.cat([labels, predicted_labels]) 

    # Split data into train, validation, and test sets
    train_embeddings, val_test_embeddings, train_labels, val_test_labels = train_test_split(all_embeddings, all_labels, test_size=0.4, random_state=42)
    val_embeddings, test_embeddings, val_labels, test_labels = train_test_split(val_test_embeddings, val_test_labels, test_size=0.5, random_state=42)

    # Create DataLoader for train, validation, and test sets
    train_dataset = CustomDataset(train_embeddings, train_labels, tokenizer, max_len=256)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    val_dataset = CustomDataset(val_embeddings, val_labels, tokenizer, max_len=256)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    test_dataset = CustomDataset(test_embeddings, test_labels, tokenizer, max_len=256)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    
    # Define optimizer and loss function for BERT model
    # optimizer = optim.Adam(model.parameters(), lr=LR)

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    # Train the BERT model using combined dataset
    # train_losses, val_losses = train_model(model, train_loader, optimizer, criterion, val_loader, EPOCHS)
    train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, EPOCHS)

    torch.save(model.state_dict(), 'classifier_model.pth')
    # Plot the training and validation loss curves
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("Classifier_loss_curve.png")  # Save the plot as loss_curve.png
    # Evaluate the trained model
    test_loss, test_accuracy, precision, recall, f1 = evaluate_model(model, criterion, test_loader)
  
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')


