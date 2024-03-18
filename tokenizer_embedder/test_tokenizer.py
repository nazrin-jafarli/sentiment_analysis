import sentencepiece as spm
from preprocess_data import preprocess_text


# Load the SentencePiece model
spm_model_path = "sp_az_tokenizer/azerbaijani_spm.model"
sp_model = spm.SentencePieceProcessor(model_file=spm_model_path)

# Define your sample sentence
sample_sentence = "Mən evə gedirəm, sağolun. Görüşənədək!"

# Preprocess the sample sentence
preprocessed_sentence = preprocess_text(sample_sentence)

# Encode the preprocessed sentence using the loaded SentencePiece model
encoded_tokens = sp_model.encode_as_pieces(preprocessed_sentence)
print(encoded_tokens)