import os
import sentencepiece as spm

# Path to the text file containing Azerbaijani language text data
azerbaijani_text_file = 'data/combined_sentences.txt'
spm_model_prefix = "sp_az_tokenizer/azerbaijani_spm"
folder_path="sp_az_tokenizer"
def train_or_load_sentencepiece_model(text_file, spm_model_prefix, vocab_size=16000, max_sentence_length=1024):
    """
    Trains a SentencePiece model from the text file if it doesn't exist, 
    otherwise loads the existing model.
    """
    # Create the folder if it doesn't exist
    folder_path = os.path.dirname(spm_model_prefix)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Check if the SentencePiece model already exists
    if os.path.exists(f"{spm_model_prefix}.model"):
        print("Found existing SentencePiece model. Loading...")
        sp_tokenizer = spm.SentencePieceProcessor()
        sp_tokenizer.load(f"{spm_model_prefix}.model")
        return sp_tokenizer

    # Otherwise, train a new SentencePiece model
    print("SentencePiece model not found. Training...")
    # Read the text data from the file
    with open(text_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    
    # Write each sentence to the training file
    spm_train_input = f"{folder_path}/azerbaijani_text.txt"
    with open(spm_train_input, "w", encoding="utf-8") as f:
        for sentence in data:
            f.write(sentence.strip() + "\n")  # Write each sentence on a separate line

    # Train SentencePiece model
    spm.SentencePieceTrainer.train(input=spm_train_input, model_prefix=spm_model_prefix, vocab_size=vocab_size, character_coverage=0.9995, max_sentence_length=max_sentence_length)

    # Load the trained SentencePiece model
    sp_tokenizer = spm.SentencePieceProcessor()
    sp_tokenizer.load(f"{spm_model_prefix}.model")

    return sp_tokenizer

print(f"========Training or loading SentencePiece model from {azerbaijani_text_file}=======")
sp_tokenizer = train_or_load_sentencepiece_model(azerbaijani_text_file, spm_model_prefix)

# Test SentencePiece tokenizer
input_string = "Mən təxminən axşam saat altıda evə qayıdacağam."
encoded_tokens = sp_tokenizer.encode_as_pieces(input_string)
print(encoded_tokens)
