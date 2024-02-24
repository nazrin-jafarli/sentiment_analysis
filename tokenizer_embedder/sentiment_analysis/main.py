import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
from transformers import BertModel
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
# Load your trained SentencePiece model
spm_model_path = "../sp_az_tokenizer/azerbaijani_spm.model"
sp = spm.SentencePieceProcessor(model_file=spm_model_path)


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
        
        
        return x.squeeze(1)  # Squeeze the redundant dimension


# Load positive and negative class data from text files
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(line.strip())
    return data

positive_data = load_data('datasets/positive/positive_data.txt')
negative_data = load_data('datasets/negative/negative_data.txt')


# Tokenize the data using SentencePiece and pad sequences
max_seq_length = 256

def process_data_with_attention(data, max_length):
    padded_data = []
    for text in data:
        tokens = sp.encode(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens += [0] * (max_length - len(tokens))
        padded_data.append(torch.tensor(tokens))
    padded_data = torch.stack(padded_data)  # Convert list of tensors to a single tensor
    attention_mask = (padded_data != 0).float()
    return padded_data, attention_mask
        

process_data_with_attention(positive_data, max_seq_length)


# # Tokenize the positive and negative data and create attention masks
positive_padded, positive_attention_mask = process_data_with_attention(positive_data, max_seq_length)
negative_padded, negative_attention_mask = process_data_with_attention(negative_data, max_seq_length)

# Generate embeddings for positive and negative class data
positive_embeddings = []
negative_embeddings = []

with torch.no_grad():
    for input_ids, attention_mask in zip(positive_padded, positive_attention_mask):
        outputs = bert_model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        last_hidden_state = outputs.last_hidden_state
        positive_embeddings.append(last_hidden_state)

    for input_ids, attention_mask in zip(negative_padded, negative_attention_mask):
        outputs = bert_model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        last_hidden_state = outputs.last_hidden_state
        negative_embeddings.append(last_hidden_state)


# Combine positive and negative embeddings and labels
all_embeddings = torch.cat(positive_embeddings + negative_embeddings)
labels = torch.tensor([1] * len(positive_embeddings) + [0] * len(negative_embeddings))


# Split data into training, validation, and test sets
train_val_embeddings, test_embeddings, train_val_labels, test_labels = train_test_split(all_embeddings, labels, test_size=0.2, random_state=42)
train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(train_val_embeddings, train_val_labels, test_size=0.25, random_state=42)

# Convert labels to one-hot encoded format
num_classes = 2

# Convert labels to one-hot encoded format
train_labels_one_hot = F.one_hot(train_labels, num_classes=num_classes)
val_labels_one_hot = F.one_hot(val_labels, num_classes=num_classes)
test_labels_one_hot = F.one_hot(test_labels, num_classes=num_classes)

# print('one hot encoded: ',train_labels_one_hot)
# Define classifier model and loss function
input_size = bert_model.config.hidden_size
hidden_size = 128

classifier_model = SimpleClassifier(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.Adam(classifier_model.parameters(), lr=0.001)

# Train the classifier
num_epochs = 3
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    optimizer.zero_grad()
    outputs = classifier_model(train_embeddings)
    
    # Apply softmax to the outputs
    outputs_softmax = F.softmax(outputs, dim=1)
    
    train_loss = criterion(outputs_softmax, train_labels_one_hot)  # Use one-hot encoded labels
     
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())

    # Validation
    with torch.no_grad():
        val_outputs = classifier_model(val_embeddings)
        # Apply softmax to the validation outputs
        val_outputs_softmax = F.softmax(val_outputs, dim=1)
        val_loss = criterion(val_outputs_softmax,val_labels_one_hot)
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
    # Apply softmax to the test outputs
    test_outputs_softmax = F.softmax(test_outputs, dim=1)
    test_loss = criterion(test_outputs_softmax, test_labels_one_hot)
    print(f'Test Loss: {test_loss.item()}')
