from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M", token=True, src_lang="azj_Latn"
)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", token=True)

# Define source and target folders
# source_folder = "main_aze_data/negative" 
# target_folder = "main_eng_data/negative"  

# source_folder = "main_aze_data/neutral" 
# target_folder = "main_eng_data/neutral" 

# source_folder = "main_aze_data/positive" 
# target_folder = "main_eng_data/positive" 

source_folder = "main_aze_data/unlabelled" 
target_folder = "main_eng_data/unlabelled" 

# Iterate over files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file is a text file
    if filename.endswith(".txt"):
        source_file_path = os.path.join(source_folder, filename)
        
        # Create corresponding target file name
        target_filename = filename.replace(".txt", "_translated.txt")
        target_file_path = os.path.join(target_folder, target_filename)
        
        # Open the source file
        with open(source_file_path, "r", encoding="utf-8") as source_file:

            sentences = source_file.readlines()

        # Translate each sentence
        translated_sentences = []
        for sentence in sentences:
            # Tokenize the sentence
            inputs = tokenizer(sentence.strip(), return_tensors="pt")

            # Generate translation
            translated_tokens = model.generate(
                **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=30
            )
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            translated_sentences.append(translation)

        # Write translated sentences to a new file in the target folder
        with open(target_file_path, "w", encoding="utf-8") as target_file:

            for translated_sentence in translated_sentences:
                target_file.write(translated_sentence + "\n")