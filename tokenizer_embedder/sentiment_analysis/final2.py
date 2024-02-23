import torch
import torch.nn as nn
import sentencepiece as spm
from transformers import BertModel
import matplotlib.pyplot as plt

# Load your trained SentencePiece model
spm_model_path = "../sp_az_tokenizer/azerbaijani_spm.model"
sp = spm.SentencePieceProcessor()
sp.load(spm_model_path)

# Load your pre-trained BERT model
bert_model_path = "../bert_mlm_az_model"
bert_model = BertModel.from_pretrained(bert_model_path)

# Freeze BERT parameters
for param in bert_model.parameters():
    param.requires_grad = False

# Define Simple Classification Network
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load positive and negative class data from text files
positive_data = []
with open('datasets/positive/positive_data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        positive_data.append(line.strip())

negative_data = []
with open('datasets/negative/negative_data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        negative_data.append(line.strip())

# Tokenize the data using SentencePiece
positive_tokenized = [sp.encode(text) for text in positive_data]
negative_tokenized = [sp.encode(text) for text in negative_data]

# Generate embeddings for positive and negative class data
positive_embeddings = []
negative_embeddings = []

with torch.no_grad():
    for text_ids in positive_tokenized:
        input_ids = torch.tensor([text_ids]).unsqueeze(0)
        embeddings = bert_model(input_ids)[0]
        positive_embeddings.append(embeddings[:, 0, :])

    for text_ids in negative_tokenized:
        input_ids = torch.tensor([text_ids]).unsqueeze(0)
        embeddings = bert_model(input_ids)[0]
        negative_embeddings.append(embeddings[:, 0, :])

# Combine positive and negative embeddings and labels
all_embeddings = torch.stack(positive_embeddings + negative_embeddings)
labels = torch.tensor([1] * len(positive_embeddings) + [0] * len(negative_embeddings))

# Shuffle data
perm = torch.randperm(all_embeddings.size(0))
all_embeddings = all_embeddings[perm]
labels = labels[perm]

# Divide data into training, validation, and test datasets
train_size = int(0.6 * len(all_embeddings))
val_size = int(0.2 * len(all_embeddings))
test_size = len(all_embeddings) - train_size - val_size

train_embeddings, val_embeddings, test_embeddings = torch.split(all_embeddings, [train_size, val_size, test_size])
train_labels, val_labels, test_labels = torch.split(labels, [train_size, val_size, test_size])

# Define classifier model and loss function
input_size = bert_model.config.hidden_size  # BERT output size
hidden_size = 256  # Adjust according to your needs
num_classes = 2  # Example: 2 classes for binary classification
classifier_model = SimpleClassifier(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)

# Train the classifier
num_epochs = 3
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    # Training
    optimizer.zero_grad()
    outputs = classifier_model(train_embeddings)
    train_loss = criterion(outputs, train_labels)
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())
    
    # Validation
    with torch.no_grad():
        val_outputs = classifier_model(val_embeddings)
        val_loss = criterion(val_outputs, val_labels)
        val_losses.append(val_loss.item())
    
    print(f'Epoch {epoch+1}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}')

# Plotting training and validation loss curves
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Evaluate on test data
with torch.no_grad():
    test_outputs = classifier_model(test_embeddings)
    test_loss = criterion(test_outputs, test_labels)
    print(f'Test Loss: {test_loss.item()}')
