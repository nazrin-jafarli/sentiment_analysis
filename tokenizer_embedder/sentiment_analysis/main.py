import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
from transformers import BertModel
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import random
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.decomposition import PCA
from scipy.special import softmax

# Load your trained SentencePiece model
spm_model_path = "../sp_az_tokenizer/azerbaijani_spm.model"
sp = spm.SentencePieceProcessor(model_file=spm_model_path)

dist_measure='euclidean' # euclidean or cosine 

# Load your pre-trained BERT model
bert_model_path = "../bert_mlm_az_model"
bert_model = BertModel.from_pretrained(bert_model_path)


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to GPU
bert_model = bert_model.to(device)

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
        # return x.squeeze(1)  # Squeeze the redundant dimension
        return x  # Output logits for each class, no need to squeeze the dimension



# Load positive and negative class data from text files
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(line.strip())
    return data

positive_data = load_data('datasets/positive/positive_data.txt')
negative_data = load_data('datasets/negative/negative_data.txt')
unlabelled_data=load_data('datasets/unlabelled/unlabelled_data.txt')

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
        

# # Tokenize the positive and negative data and create attention masks
positive_padded, positive_attention_mask = process_data_with_attention(positive_data, max_seq_length)
positive_padded = positive_padded.to(device)
positive_attention_mask = positive_attention_mask.to(device)

negative_padded, negative_attention_mask = process_data_with_attention(negative_data, max_seq_length)
negative_padded = negative_padded.to(device)
negative_attention_mask = negative_attention_mask.to(device)


unlabeled_padded, unlabeled_attention_mask = process_data_with_attention(unlabelled_data, max_seq_length)
unlabeled_padded=unlabeled_padded.to(device)
unlabeled_attention_mask=unlabeled_attention_mask.to(device)



# Generate embeddings for positive and negative class data
positive_embeddings = []
negative_embeddings = []
unlabeled_embeddings=[]


# Generate embeddings for positive and negative class data
def generate_embeddings(padded_data, attention_mask):
    embeddings = []
    with torch.no_grad():
        for input_ids, mask in zip(padded_data, attention_mask):
            outputs = bert_model(input_ids.unsqueeze(0), attention_mask=mask.unsqueeze(0))
            last_hidden_state = outputs.last_hidden_state
            embeddings.append(last_hidden_state.squeeze(0))
    return torch.stack(embeddings)


positive_embeddings = generate_embeddings(positive_padded, positive_attention_mask)
negative_embeddings = generate_embeddings(negative_padded, negative_attention_mask)
unlabeled_embeddings = generate_embeddings(unlabeled_padded, unlabeled_attention_mask)

# Combine positive and negative embeddings and labels
labeled_embeddings = torch.cat([positive_embeddings, negative_embeddings])
labels = torch.cat([torch.ones(len(positive_embeddings)), torch.zeros(len(negative_embeddings))])
print('labels:',type(labels))


class KMeansSemiSupervised:
    def __init__(self, n_clusters=2, max_iter=10, distance_measure='cosine'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.distance_measure = distance_measure
        self.unlabeled_embeddings = None
        self.centroid_labels = None  # Add centroid_labels attribute


    def initialize_centroids(self, labeled_data, labels):
        if len(labeled_data) > 0:
            self.centroids = {}
            self.centroid_labels = {}
            unique_labels = torch.unique(labels)
            for idx, label_value in enumerate(unique_labels):
                label_data = labeled_data[labels == label_value]
                centroid = label_data.mean(dim=0)
                self.centroids[idx] = centroid
                self.centroid_labels[idx] = label_value.item()
        else:
            self.initialize_centroids_randomly()


    def initialize_centroids_randomly(self):
        random_indices = torch.randperm(len(self.unlabeled_embeddings))[:self.n_clusters]
        self.centroids = {i: self.unlabeled_embeddings[index] for i, index in enumerate(random_indices)}
        self.centroid_labels = {i: None for i in range(self.n_clusters)}

    def fit(self, labeled_embeddings, unlabeled_embeddings, labels):
        self.unlabeled_embeddings = unlabeled_embeddings
        self.initialize_centroids(labeled_embeddings, labels)

        # print(self.distance_measure)

        for _ in range(self.max_iter):
            nearest_labeled_centroid_indices = []
            for x in unlabeled_embeddings:
                distances = []
                for centroid in self.centroids.values():
                    if self.distance_measure == 'cosine':
                        similarity = torch.dot(x, centroid) / (torch.norm(x) * torch.norm(centroid))
                        distance = 1.0 - similarity
                    elif self.distance_measure == 'euclidean':
                        distance = torch.dist(x, centroid)
                    distances.append(distance)
                nearest_labeled_centroid_indices.append(torch.argmin(torch.tensor(distances)).item())

            nearest_labeled_centroid_indices = torch.tensor(nearest_labeled_centroid_indices)

            for cluster_idx in range(self.n_clusters):
                cluster_indices = (nearest_labeled_centroid_indices == cluster_idx).nonzero().flatten()
                cluster_embeddings = unlabeled_embeddings[cluster_indices]
                if len(cluster_embeddings) > 0:
                    self.centroids[cluster_idx] = cluster_embeddings.mean(dim=0)

        return self.centroids

    def predict(self, unlabeled_embeddings):
        cluster_indices = []
        for x in unlabeled_embeddings:
            distances = []
            for centroid in self.centroids.values():
                if self.distance_measure == 'cosine':
                    similarity = torch.dot(x, centroid) / (torch.norm(x) * torch.norm(centroid))
                    distance = 1.0 - similarity
                elif self.distance_measure == 'euclidean':
                    distance = torch.dist(x, centroid)
                distances.append(distance)
            closest_centroid_label = min(self.centroids, key=lambda c: distances[c])
            semantic_label = self.centroid_labels[closest_centroid_label]
            cluster_indices.append(semantic_label)
        return cluster_indices


# Initialize KMeansSemiSupervised instance
kmeans_semi_supervised = KMeansSemiSupervised(n_clusters=2, max_iter=100,distance_measure=dist_measure) #distance_measure 'cosine' or 'euclidean' 

# Train your model and obtain the trained centroids
kmeans_semi_supervised.fit(labeled_embeddings, unlabeled_embeddings, labels)

# Use the trained centroids to predict the labels of your unlabeled data
predicted_labels = kmeans_semi_supervised.predict(unlabeled_embeddings)
predicted_labels=torch.tensor(predicted_labels)
print('labels:',labels)
print(type(labels))
print('predicted_labels',predicted_labels)
print(type(predicted_labels))

num_clusters = kmeans_semi_supervised.n_clusters

# Combine all data for splitting
all_embeddings = torch.cat([labeled_embeddings, unlabeled_embeddings])
all_labels = torch.cat([labels, predicted_labels]) 


# Split combined data and labels
train_val_embeddings, test_embeddings, train_val_labels, test_labels = train_test_split(
    all_embeddings,
    all_labels,
    test_size=0.2,
    random_state=42
)

# Split train-val data further
train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
    train_val_embeddings,
    train_val_labels,
    test_size=0.25,
    random_state=42
)

# Ensure labels are tensors before one-hot encoding (training only)
train_labels = train_labels.long()  # Convert to long for one-hot encoding
val_labels = val_labels.long()
test_labels=test_labels.long()


# Convert training labels to one-hot encoded format
num_classes = 2

train_labels_one_hot = F.one_hot(train_labels, num_classes=num_classes)
train_labels_one_hot = train_labels_one_hot.to(device)

val_labels_one_hot = F.one_hot(val_labels, num_classes=num_classes)
val_labels_one_hot=val_labels_one_hot.to(device)

test_labels_one_hot = F.one_hot(test_labels, num_classes=num_classes)
test_labels_one_hot=test_labels_one_hot.to(device)

# Define classifier model and loss function
input_size = bert_model.config.hidden_size  # Include cluster information size
hidden_size = 128

classifier_model = SimpleClassifier(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
# Move criterion and optimizer to GPU
criterion = criterion.to(device)
classifier_model = classifier_model.to(device)
# Define optimizer
optimizer = optim.Adam(classifier_model.parameters(), lr=0.001)


# Train the classifier
num_epochs = 3
train_losses = []
val_losses = []


for epoch in range(num_epochs):
    # Training
    optimizer.zero_grad()
    # Forward pass
    outputs = classifier_model(train_embeddings)
    # Compute loss using combined outputs and labels
    train_loss = criterion(outputs, train_labels_one_hot)
    
    # Backpropagation
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())

    # Validation
    with torch.no_grad():
        val_outputs = classifier_model(val_embeddings)  # Pass validation embeddings
        val_loss = criterion(val_outputs, val_labels_one_hot)  # Compute validation loss
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




