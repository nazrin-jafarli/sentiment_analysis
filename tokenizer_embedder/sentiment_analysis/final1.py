import torch
import torch.nn as nn
import sentencepiece as spm
from transformers import BertModel
# Load your trained SentencePiece model
spm_model_path = "sp_az_tokenizer/azerbaijani_spm.model"
sp = spm.SentencePieceProcessor()
sp.load(spm_model_path)

# Assuming you have a pre-trained BERT model saved as 'bert_model.pt'
# Load your pre-trained BERT model
bert_model_path = "bert_mlm_az_model"
bert_model = BertModel.from_pretrained(bert_model_path)

# Freeze BERT parameters
for param in bert_model.parameters():
    param.requires_grad = False



# Step 1: Define Simple Classification Network
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

positive_data = []
with open('datasets/positive_data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        positive_data.append(line.strip())

negative_data = []
with open('datasets/negative_data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        negative_data.append(line.strip())


# # Load positive and negative class data from text files
# positive_data = ...
# negative_data = ...

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


num_epochs=3
# Step 3: Define and forward through the Simple Classifier
input_size = bert_model.config.hidden_size  # BERT output size
hidden_size = 256  # Adjust according to your needs
num_classes = 2  # Example: 2 classes for binary classification
classifier_model = SimpleClassifier(input_size, hidden_size, num_classes)
# output = classifier_model(sentence_embedding)




# Define classifier model and loss function
classifier_model = SimpleClassifier(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)

# Train the classifier
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = classifier_model(all_embeddings)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
