import torch
from transformers import BertModel
import sentencepiece as spm

# Load pre-trained BERT model
model_name = "bert_mlm_az_model"
model = BertModel.from_pretrained(model_name)

# Load SentencePiece model
spm_model_path = "sp_az_tokenizer/azerbaijani_spm.model"
sp_model = spm.SentencePieceProcessor(model_file=spm_model_path)

# Define your sentence
sentence = "Mən evə gedirəm,sağolun."

# Convert tokens to token IDs
indexed_tokens = sp_model.encode(sentence, out_type=int)
print(indexed_tokens)
# Convert token IDs to tensor
tokens_tensor = torch.tensor([indexed_tokens])

# Get model embeddings
with torch.no_grad():
    outputs = model(tokens_tensor)
    # `outputs` is a tuple containing various elements, including hidden states
    # For the embedding of the last layer, you typically access the last element of `outputs`
    # You can also choose a specific layer's embeddings by indexing into `outputs` accordingly
    last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, seq_length, hidden_size]

# Now `last_hidden_state` contains the embeddings of each token in the input sentence
# You can use these embeddings for downstream tasks or analysis as needed
print(last_hidden_state)
print(last_hidden_state.shape)