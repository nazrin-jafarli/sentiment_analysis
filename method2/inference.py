from transformers import BertTokenizer, BertForSequenceClassification
import torch
from nllb200_inference import translate_with_nnlb200
# Load your pre-trained model's state_dict
model_state_dict = torch.load("classifier_model.pth")  # Replace "classifier_model.pth" with the path to your saved model

# Instantiate the model architecture without weights
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, output_hidden_states=True)

# Load the pre-trained weights into the model
model.load_state_dict(model_state_dict)

# Set the model to evaluation mode
model.eval()

# Instantiate the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def run_inference(text):
    # Tokenize input text
    text = text.lower()
    print(text)
    text=translate_with_nnlb200(text)
    print(text)

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted label
    _, predicted = torch.max(outputs.logits, 1)
    predicted_label = predicted.item()  # Assuming batch size is 1
    # Map numerical label to descriptive label
    label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_label = label_mapping[predicted_label]
    return predicted_label

# Example usage
# text = "The positive testimonials we receive from our clients are a source of motivation to constantly seek excellence in cybersecurity."
# predicted_label = run_inference(text)
# print("Predicted Label:", predicted_label)
