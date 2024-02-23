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

# Step 2: Tokenize input text and get embeddings
input_text = "Your Azerbaijani input text goes here."
# Tokenize input text using SentencePiece
input_ids = torch.tensor(sp.encode(input_text)).unsqueeze(0)
with torch.no_grad():
    embeddings = bert_model(input_ids)

# Extract the embeddings from BERT (you might need to adjust this based on your BERT model's output)
sentence_embedding = embeddings[:, 0, :]  # Assuming you want to use the embedding of the [CLS] token

# Step 3: Define and forward through the Simple Classifier
input_size = bert_model.config.hidden_size  # BERT output size
hidden_size = 256  # Adjust according to your needs
num_classes = 2  # Example: 2 classes for binary classification
classifier_model = SimpleClassifier(input_size, hidden_size, num_classes)
output = classifier_model(sentence_embedding)

# Step 4: Print or use the output for further processing
print(output)
