
import os
import sentencepiece as spm

# Path to the text file containing Azerbaijani language text data
azerbaijani_text_file = 'main_data/final_data_10000.txt'
spm_model_prefix = "SP_aze_tokenizer/azerbaijani_spm"
folder_path = "SP_aze_tokenizer"

def train_sentencepiece_model(text_file, spm_model_prefix, vocab_size=16000, max_sentence_length=512):
    """
    Trains a SentencePiece model from the text file if it doesn't exist, 
    otherwise loads the existing model.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define special tokens
    special_tokens = "[CLS],[SEP],[MASK]"

    # Train SentencePiece model using Byte Pair Encoding (BPE)
    spm.SentencePieceTrainer.train(input=text_file, model_prefix=spm_model_prefix, vocab_size=vocab_size, character_coverage=0.9995, max_sentence_length=max_sentence_length, control_symbols=special_tokens, model_type="bpe")

    # Load the trained SentencePiece model
    sp_tokenizer = spm.SentencePieceProcessor()
    sp_tokenizer.load(f"{spm_model_prefix}.model")

    return sp_tokenizer

print(f"========Training SentencePiece model from {azerbaijani_text_file}=======")
train_sentencepiece_model(azerbaijani_text_file, spm_model_prefix)
