import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
from transformers import BertModel, BertForMaskedLM
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
# from torch.nn.utils.rnn import pad_sequence  # I just did manual padding 
# from sklearn.metrics import accuracy_score # I calculated accuracy manually
import numpy as np
# from sklearn.decomposition import PCA  # can be needed for k clustering visualization
# from scipy.special import softmax # no need as nn.CrossEntropyLoss() internally applies softmax activation to the logits
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from kmeans import KMeansSemiSupervised

# Load your trained SentencePiece model
spm_model_path = "../tokenizer_embedder/SP_aze_tokenizer/azerbaijani_spm.model"
sp = spm.SentencePieceProcessor(model_file=spm_model_path)

dist_measure='euclidean' # euclidean or cosine 

# Load your pre-trained BERT model
bert_model_path = "../tokenizer_embedder/BertMasked_aze_embedder"
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


# Function to calculate evaluation metrics for multi-class classification
def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    return precision, recall, f1


# Visualize the confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    # Convert one-hot encoded labels back to original class labels
    true_labels_argmax = np.argmax(true_labels, axis=1)
    predicted_labels_argmax = np.argmax(predicted_labels, axis=1)

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels_argmax, predicted_labels_argmax)
    
    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.legend()
    plt.savefig("Confusion Matrix.png")  # Save the plot as loss_curve.png
    # plt.show()


# Load positive and negative class data from text files
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(line.strip())
    return data

positive_data = load_data('./main_data/positive/final_positive_aze.txt')
negative_data = load_data('./main_data/negative/final_negative_aze.txt')
neutral_data = load_data('./main_data/neutral/final_neutral_aze.txt')
unlabelled_data=load_data('./main_data/unlabelled/final_unlabelled_aze.txt')

# Tokenize the data using SentencePiece and pad sequences
max_seq_length = 256

def process_data_with_attention(data, max_length):
    padded_data = []
    for text in data:
        # Lowercase the text before tokenization
        text = text.lower()
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
print('positive_padded',positive_padded)
positive_padded = positive_padded.to(device)
positive_attention_mask = positive_attention_mask.to(device)

negative_padded, negative_attention_mask = process_data_with_attention(negative_data, max_seq_length)
negative_padded = negative_padded.to(device)
negative_attention_mask = negative_attention_mask.to(device)

neutral_padded, neutral_attention_mask = process_data_with_attention(neutral_data, max_seq_length)
neutral_padded = neutral_padded.to(device)
neutral_attention_mask = neutral_attention_mask.to(device)


unlabeled_padded, unlabeled_attention_mask = process_data_with_attention(unlabelled_data, max_seq_length)
unlabeled_padded=unlabeled_padded.to(device)
unlabeled_attention_mask=unlabeled_attention_mask.to(device)



# Generate embeddings for positive and negative class data
positive_embeddings = []
negative_embeddings = []
neutral_embeddings=[]
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
print('positive_embeddings',positive_embeddings)
negative_embeddings = generate_embeddings(negative_padded, negative_attention_mask)
neutral_embeddings = generate_embeddings(neutral_padded, neutral_attention_mask)
unlabeled_embeddings = generate_embeddings(unlabeled_padded, unlabeled_attention_mask)

# Combine positive and negative embeddings and labels
labeled_embeddings = torch.cat([positive_embeddings, negative_embeddings,neutral_embeddings])
# Assign labels: 0 for negative, 1 for positive, and 2 for neutral
labels = torch.cat([torch.ones(len(positive_embeddings)), torch.zeros(len(negative_embeddings)), torch.full((len(neutral_embeddings),), 2)])
print('labels:', type(labels))

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
num_classes = 3

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


# Define DataLoader for batch-wise training
batch_size = 8  # Define your desired batch size
train_data = torch.utils.data.TensorDataset(train_embeddings.to(device), train_labels_one_hot)
print("train_data",train_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
print("train_loader",train_loader)

val_data = torch.utils.data.TensorDataset(val_embeddings.to(device), val_labels_one_hot)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_data = torch.utils.data.TensorDataset(test_embeddings.to(device), test_labels_one_hot)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


# Function to train the classifier and collect losses
def train_classifier(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # Initialize with a very high value
    best_model_state = None  # Initialize to None
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            batch_outputs = model(batch_data)
            batch_loss = criterion(batch_outputs, batch_labels)
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
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_outputs = model(batch_data)
                batch_loss = criterion(batch_outputs, batch_labels)
                epoch_val_loss += batch_loss.item()
                _, predicted = torch.max(batch_outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)


        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')
   
    # Load the best model state before returning
    model.load_state_dict(best_model_state)
    return train_losses, val_losses



# Train the classifier and collect losses
train_losses, val_losses = train_classifier(classifier_model, criterion, optimizer, train_loader, val_loader, num_epochs)

# Save the trained model
torch.save(classifier_model.state_dict(), 'classifier_model.pth')

# Plot the training and validation loss curves
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("Classifier_loss_curve.png")  # Save the plot as loss_curve.png
# plt.show()

# Evaluate on test data
def evaluate_model(model, criterion, test_loader):
    test_loss = 0
    test_total = 0
    test_correct = 0
    predicted_labels = []
    true_labels = []
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_outputs = model(batch_data)
            batch_loss = criterion(batch_outputs, batch_labels)
            test_loss += batch_loss.item()
            _, predicted = torch.max(batch_outputs.data, 1)
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(batch_labels.cpu().numpy())
    test_loss /= len(test_loader)
    test_accuracy = test_correct / test_total
    return test_loss, test_accuracy, np.array(true_labels), np.array(predicted_labels)


# early_stopping_patience = 3
train_classifier(classifier_model, criterion, optimizer, train_loader, val_loader, num_epochs)

# Evaluate the trained model
test_loss, test_accuracy, true_labels, predicted_labels = evaluate_model(classifier_model, criterion, test_loader)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Calculate additional evaluation metrics
precision, recall, f1 = calculate_metrics(true_labels, predicted_labels)
print(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}')

# Visualize the confusion matrix
plot_confusion_matrix(true_labels, predicted_labels, classes=['Class 0', 'Class 1','Class 2'])


# Save evaluation metrics to a file
with open("evaluation_metrics.txt", "w") as file:
    file.write(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}\n')
    file.write(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}\n')