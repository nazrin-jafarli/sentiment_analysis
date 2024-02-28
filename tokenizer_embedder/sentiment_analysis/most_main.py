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

# Load your trained SentencePiece model
spm_model_path = "../sp_az_tokenizer/azerbaijani_spm.model"
sp = spm.SentencePieceProcessor(model_file=spm_model_path)


# Load your pre-trained BERT model
bert_model_path = "../bert_mlm_az_model"
bert_model = BertModel.from_pretrained(bert_model_path)


# Freeze BERT parameters
for param in bert_model.parameters():
    param.requires_grad = False




class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, cluster_labels):
        print("Shapes:")
        print("x:", x.shape)
        print("cluster_labels:", cluster_labels.shape)
        
        x = torch.cat([x, cluster_labels.unsqueeze(1)], dim=1)  # Concatenate input and cluster labels
        print("Concatenated shape:", x.shape)        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(1)  # Squeeze redundant dimension



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
        

# process_data_with_attention(positive_data, max_seq_length)


# # Tokenize the positive and negative data and create attention masks
positive_padded, positive_attention_mask = process_data_with_attention(positive_data, max_seq_length)
negative_padded, negative_attention_mask = process_data_with_attention(negative_data, max_seq_length)
unlabelled_padded, unlabelled_attention_mask = process_data_with_attention(unlabelled_data, max_seq_length)

# Generate embeddings for positive and negative class data
positive_embeddings = []
negative_embeddings = []



# Generate embeddings for positive and negative class data
def generate_embeddings(padded_data, attention_mask):
    embeddings = []
    with torch.no_grad():
        for input_ids, mask in zip(padded_data, attention_mask):
            outputs = bert_model(input_ids.unsqueeze(0), attention_mask=mask.unsqueeze(0))
            last_hidden_state = outputs.last_hidden_state
            embeddings.append(last_hidden_state.squeeze(0))
    return torch.stack(embeddings)

print('positive_padded',positive_padded.shape)
print('positive_attention_mask',positive_attention_mask.shape)
print('negative_padded',negative_padded.shape)
print('negative_attention_mask',negative_attention_mask.shape)

positive_embeddings = generate_embeddings(positive_padded, positive_attention_mask)
negative_embeddings = generate_embeddings(negative_padded, negative_attention_mask)
unlabelled_embeddings = generate_embeddings(unlabelled_padded, unlabelled_attention_mask)
print('positive_embeddings',positive_embeddings.shape)
print('negative_embeddings',negative_embeddings.shape)
print('unlabelled_embeddings',unlabelled_embeddings.shape)
# Combine positive and negative embeddings and labels
all_embeddings = torch.cat([positive_embeddings, negative_embeddings])
labels = torch.cat([torch.ones(len(positive_embeddings)), torch.zeros(len(negative_embeddings))])

class KMeansSemiSupervised:
    def __init__(self, n_clusters=2, max_iter=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.prev_centroids = None 

    def initialize_centroids(self, labeled_data, labeled_labels):
        """
        Initialize centroids using labeled data.
        """
        if len(labeled_data) > 0:
            unique_labels = torch.unique(labeled_labels)
            initial_centroids = []
            for label in unique_labels:
                label_data = labeled_data[labeled_labels == label]
                centroid = torch.mean(label_data, dim=0)
                initial_centroids.append(centroid)
            if len(initial_centroids) < self.n_clusters:
                initial_centroids += [labeled_data[0]] * (self.n_clusters - len(initial_centroids))
            self.centroids = {i: centroid for i, centroid in enumerate(initial_centroids)}
            self.prev_centroids = {i: centroid for i, centroid in enumerate(initial_centroids)}
        else:
            self.initialize_centroids_randomly(labeled_data)

    def initialize_centroids_randomly(self, labeled_data):
        random_indices = torch.randperm(len(labeled_data))[:self.n_clusters]
        self.centroids = {i: labeled_data[index] for i, index in enumerate(random_indices)}
        self.prev_centroids = {}

    def fit(self, labeled_data, unlabeled_data, labeled_labels):
        num_labeled_data = len(labeled_labels)
        self.initialize_centroids(labeled_data, labeled_labels)
        iteration = 0
        converged = False

        while not converged and iteration < self.max_iter:
            cluster_points = {label: [] for label in self.centroids}
            # Convert centroids from dictionary to tensor
            centroids_tensor = torch.stack(list(self.centroids.values()))

            # Ensure labeled_data is a 2D tensor
            labeled_data_2d = labeled_data.view(labeled_data.size(0), -1)

            # Ensure centroids_tensor is a 2D tensor
            centroids_tensor_2d = centroids_tensor.view(centroids_tensor.size(0), -1)

            # Assign labeled data points to nearest centroids using cosine similarity
            labeled_dists = sk_cosine_similarity(labeled_data_2d, centroids_tensor_2d)
            labeled_closest_centroids_idx = torch.argmax(torch.tensor(labeled_dists), dim=1)

            for x, centroid_idx in zip(labeled_data, labeled_closest_centroids_idx):
                cluster_points[centroid_idx.item()].append(x)
            
            print('Shape of unlabeled_embeddings:', unlabeled_data.shape)

            unlabeled_data_flattened = unlabeled_data.view(unlabeled_data.size(0), -1)

            # Assign unlabeled data points to nearest centroids using cosine similarity
            unlabeled_dists = sk_cosine_similarity(unlabeled_data_flattened, centroids_tensor_2d)
            unlabeled_closest_centroids_idx = torch.argmax(torch.tensor(unlabeled_dists), dim=1)

            for x, centroid_idx in zip(unlabeled_data, unlabeled_closest_centroids_idx):
                cluster_points[centroid_idx.item()].append(x)

            # Update centroids
            self.prev_centroids = self.centroids.copy()
            for label, cluster in cluster_points.items():
                if cluster:
                    self.centroids[label] = torch.mean(torch.stack(cluster), dim=0)

            # Check convergence
            converged = all(torch.linalg.norm(self.centroids[i] - self.prev_centroids[i]) == 0 for i in self.centroids)

            iteration += 1

        print(f"KMeans converged after {iteration} iterations.")





    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        if X.dim() == 1:
            X = X.unsqueeze(0)

        centroids_tensor = torch.stack(list(self.centroids.values()))
        
        # Reshape X and centroids_tensor to 2D tensors
        X_flattened  = X.view(X.size(0), -1)  # Flatten each sequence into a single vector
        centroids_flattened  = centroids_tensor.view(centroids_tensor.size(0), -1)  # Flatten each centroid into a single vector

        # Calculate cosine similarity
        dists = sk_cosine_similarity(X_flattened, centroids_flattened)
        print('X_2d',X_flattened.shape)
        print('centroids_tensor_2d',centroids_flattened.shape)
        print('dists',dists)
        # Convert dists to PyTorch tensor
        dists_tensor = torch.tensor(dists)
        # Get closest centroid indices
        closest_centroids_idx = torch.argmax(dists_tensor, dim=1)

        # Retrieve cluster centers for those indices (optional)
        class_centers = centroids_tensor[closest_centroids_idx]

        return closest_centroids_idx, class_centers



# Initialize KMeansSemiSupervised instance
kmeans_semi_supervised = KMeansSemiSupervised(n_clusters=2, max_iter=100)

# Prepare data for KMeans
labeled_data = torch.cat([positive_embeddings, negative_embeddings])
labeled_labels = torch.cat([torch.ones(len(positive_embeddings)), torch.zeros(len(negative_embeddings))])

# Fit K-means to labeled and unlabeled embeddings
kmeans_semi_supervised.fit(labeled_data, unlabelled_embeddings, labeled_labels)
print('labeled_data shape:', labeled_data.shape)
print('unlabelled_embeddings shape:', unlabelled_embeddings.shape)


# Predict clusters for all embeddings
class_centers, cluster_assignments = kmeans_semi_supervised.predict(all_embeddings)

# Convert cluster assignments to one-hot encoded format
num_clusters = kmeans_semi_supervised.n_clusters
cluster_labels_one_hot = F.one_hot(torch.tensor(cluster_assignments), num_classes=num_clusters)


# Split data into training, validation, and test sets
train_val_embeddings, test_embeddings, train_val_labels, test_labels = train_test_split(all_embeddings, labels, test_size=0.2, random_state=42)
train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(train_val_embeddings, train_val_labels, test_size=0.25, random_state=42)
print('train_embeddings',train_embeddings.shape)
# Ensure labels are tensors before one-hot encoding
train_labels = train_labels.long()  # Convert to long for one-hot encoding
val_labels = val_labels.long()
test_labels = test_labels.long()

# Convert labels to one-hot encoded format
num_classes = 2

# Convert labels to one-hot encoded format
train_labels_one_hot = F.one_hot(train_labels, num_classes=num_classes)
val_labels_one_hot = F.one_hot(val_labels, num_classes=num_classes)
test_labels_one_hot = F.one_hot(test_labels, num_classes=num_classes)

# print('one hot encoded: ',train_labels_one_hot)
# Define classifier model and loss function
input_size = bert_model.config.hidden_size + num_clusters  # Include cluster information size
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
    outputs = classifier_model(train_embeddings, cluster_labels_one_hot[train_labels].float())  # Pass cluster info    
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
