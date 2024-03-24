
import torch
import sentencepiece as spm
from transformers import BertModel
from train import SimpleClassifier

# Load your trained SentencePiece model
spm_model_path = "../tokenizer_embedder/SP_aze_tokenizer/azerbaijani_spm.model"
sp = spm.SentencePieceProcessor(model_file=spm_model_path)

# Load your pre-trained BERT model
bert_model_path = "../tokenizer_embedder/BertMasked_aze_embedder"
bert_model = BertModel.from_pretrained(bert_model_path)


input_size = bert_model.config.hidden_size  # Update with the appropriate input size
hidden_size = 128  # Update with the appropriate hidden size
num_classes = 3  # Update with the appropriate number of classes

# Optional: Define a mapping from class indices to labels (if applicable)
class_labels = {1: "Positive", 0: "Negative", 2: "Neutral"}  # Example mapping

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)

classifier_model = SimpleClassifier(input_size, hidden_size, num_classes)
classifier_model.load_state_dict(torch.load('classifier_model.pth'))
classifier_model = classifier_model.to(device)
# Set the classifier model to evaluation mode
classifier_model.eval()


def preprocess_text_for_inference(sentence):
    # Preprocess the input sentence
    processed_sentence = sentence.lower()
    
    # Tokenize the sentence using SentencePiece
    tokens = sp.encode(processed_sentence)
    # Pad or truncate tokens to match max sequence length
    max_seq_length = 256
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]
    else:
        tokens += [0] * (max_seq_length - len(tokens))
    
    # Convert tokens to tensor
    input_ids = torch.tensor(tokens).to(device)
    
    # Create attention mask
    attention_mask = (input_ids != 0).float().to(device)
    
    return input_ids, attention_mask

def inference(sentence):
    # Preprocess and tokenize the input sentence
    input_tensor, attention_mask_tensor = preprocess_text_for_inference(sentence)
    
    # Generate the BERT embeddings for the input sentence
    with torch.no_grad():
        outputs = bert_model(input_tensor.unsqueeze(0), attention_mask=attention_mask_tensor.unsqueeze(0))
        
        last_hidden_state = outputs.last_hidden_state
    
        # Forward pass through classifier model
        logits = classifier_model(last_hidden_state)
        print('logits',logits)

        # # Apply softmax activation function to convert logits to probabilities
        # probabilities = torch.softmax(logits, dim=1)
        # print('probabilities',probabilities.shape)
        
        # # Get the predicted class label by finding the index of the maximum probability
        # predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # return predicted_class
        
        _, predicted = torch.max(logits.data, 1)
        
        
        return predicted


# Example usage:
sentence = "korporativ təlim və tövsiyələr, işçilərin siber təhlükələrə qarşı daha çox ehtiyatlı olmasına kömək edir."

# Perform inference
# Perform inference
predicted_class = inference(sentence)
print("Predicted class:", predicted_class)