import os
import sentencepiece as spm
import re
from preprocess_data import preprocess_text

# Path to the text file containing Azerbaijani language text data
azerbaijani_text_file = 'main_data/final_data_10000.txt'
spm_model_prefix = "SP_aze_tokenizer/azerbaijani_spm"
folder_path="SP_aze_tokenizer"


def train_sentencepiece_model(text_file, spm_model_prefix, vocab_size=16000, max_sentence_length=512):
    """
    Trains a SentencePiece model from the text file if it doesn't exist, 
    otherwise loads the existing model.
    """
    # Create the folder if it doesn't exist
    folder_path = os.path.dirname(spm_model_prefix)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Read the text data from the file
    with open(text_file, 'r', encoding='utf-8') as f:
        # data = f.readlines()
        data = [preprocess_text(sentence) for sentence in f.readlines()]

    
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

print(f"========Training SentencePiece model from {azerbaijani_text_file}=======")
train_sentencepiece_model(azerbaijani_text_file, spm_model_prefix)

