import sentencepiece as spm

# Load the SentencePiece model
spm_model_path = "SP_aze_tokenizer/azerbaijani_spm.model"
sp_model = spm.SentencePieceProcessor(model_file=spm_model_path)

# Define your sample sentence
sample_sentence = "Mən evə gedirəm, sağolun. Görüşənədək!"
sample_sentence=sample_sentence.lower()

# Encode the preprocessed sentence using the loaded SentencePiece model
encoded_tokens = sp_model.encode_as_pieces(sample_sentence)
print(encoded_tokens)