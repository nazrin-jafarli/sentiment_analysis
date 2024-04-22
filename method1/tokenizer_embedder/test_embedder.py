import torch
from transformers import BertForMaskedLM 
import sentencepiece as spm

# Load SentencePiece model
spm_model_path = "SP_aze_tokenizer/azerbaijani_spm.model"
sp_model = spm.SentencePieceProcessor(model_file=spm_model_path)

# Load the trained BERT model
bert_model_path = "BertMasked_aze_embedder"
bert_model = BertForMaskedLM.from_pretrained(bert_model_path, output_hidden_states=True)

# Tokenize input text and create attention masks
def tokenize_and_encode(text, sp_model, tokenizer, max_length):
    text = text.lower()  # Lowercase the input text
    tokens = sp_model.encode(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens += [0] * (max_length - len(tokens))
    return torch.tensor(tokens).unsqueeze(0), torch.tensor([1] * len(tokens)).unsqueeze(0)

# Define a function to extract hidden states
def extract_hidden_states(outputs, extraction_type):
    if extraction_type == 'last_hidden':
        return outputs.hidden_states[-1]
    elif extraction_type == 'average':
        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states)
        layerwise_hidden_states = torch.stack(hidden_states)  # Stack hidden states along a new dimension
        return torch.mean(layerwise_hidden_states, dim=0)  # Take the mean along the new dimension
    else:
        raise ValueError("Invalid extraction type. Please use 'last_hidden' or 'average'.")


# Define your input text
input_text = "Mən evə gedirəm, sağolun. Görüşənədək!"

# Tokenize and encode input text
input_ids, attention_mask = tokenize_and_encode(input_text, sp_model, None, max_length=128)

# Forward pass through the BERT model
with torch.no_grad():
    outputs = bert_model(input_ids, attention_mask=attention_mask)


# Extract hidden states based on extraction type
# 'average' or 'last_hidden'
extraction_type = 'last_hidden' 
hidden_states = extract_hidden_states(outputs, extraction_type)

# Now 'hidden_states' contains either the last hidden state or the average of all hidden states across layers
print(hidden_states)


