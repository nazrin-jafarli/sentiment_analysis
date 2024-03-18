import json

def extract_sentences_from_json(json_file):
    """
    Extract sentences from JSON file.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences = [entry['sentence'] for entry in data['response']]
    return sentences

def combine_sentences(json_files):
    """
    Combine sentences from multiple JSON files.
    """
    all_sentences = []
    for file in json_files:
        sentences = extract_sentences_from_json(file)
        all_sentences.extend(sentences)
    return all_sentences

def write_combined_sentences_to_json(sentences, output_file):
    """
    Write combined sentences to a new JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, ensure_ascii=False, indent=4)
    print("Combined sentences have been written to:", output_file)

def write_combined_sentences_to_txt(sentences, output_file):
    """
    Write combined sentences to a new TXT file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
    print("Combined sentences have been written to:", output_file)

def calculate_vocabulary(sentences):
    """
    Calculate vocabulary size from the combined sentences and return unique tokens.
    """
    unique_tokens = set()
    for sentence in sentences:
        tokens = sentence.split()
        unique_tokens.update(tokens)
    return len(unique_tokens), unique_tokens

def write_vocabulary_to_file(unique_tokens, output_file):
    """
    Write unique tokens to a file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for token in sorted(unique_tokens):
            f.write(token + '\n')
    print("Unique tokens have been written to:", output_file)

# List of JSON files
json_files = ['azerbaijani.json']

# Combine sentences from multiple JSON files
combined_sentences = combine_sentences(json_files)

# Write combined sentences into a new JSON file
combined_json_file = 'sample2.json'
write_combined_sentences_to_json(combined_sentences, combined_json_file)

# Write combined sentences into a new TXT file
combined_txt_file = 'sample2.txt'
write_combined_sentences_to_txt(combined_sentences, combined_txt_file)

# Print count of sentences
print("Total combined sentences:", len(combined_sentences))

# Calculate vocabulary size and get unique tokens
vocabulary_size, unique_tokens = calculate_vocabulary(combined_sentences)
print("Vocabulary Size:", vocabulary_size)

# Write unique tokens to a file
vocabulary_file = 'vocabulary2.txt'
write_vocabulary_to_file(unique_tokens, vocabulary_file)
