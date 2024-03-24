# List of input file paths
input_file_paths = ['./main_data/negative/final_negative_aze.txt',
                    './main_data/positive/final_positive_aze.txt',
                    './main_data/neutral/final_neutral_aze.txt',
                    './main_data/unlabelled/final_unlabelled_aze.txt']

# Iterate over each input file path
for input_file_path in input_file_paths:
    # Open the input file in read mode
    with open(input_file_path, 'r', encoding='utf-8') as file:
        # Read the text from the file
        text = file.read()

    # Convert the text to lowercase
    lowercase_text = text.lower()

    # Write the lowercase text back to the same file
    with open(input_file_path, 'w', encoding='utf-8') as file:
        file.write(lowercase_text)

    print(f"Lowercasing completed and saved to {input_file_path}.")
