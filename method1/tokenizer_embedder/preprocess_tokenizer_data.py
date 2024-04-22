
# Open the input file in read mode
input_file_path = './main_data/final_data_10000.txt'
with open(input_file_path, 'r', encoding='utf-8') as file:
    # Read the text from the file
    text = file.read()

# Convert the text to lowercase
lowercase_text = text.lower()

# Write the lowercase text back to the same file
with open(input_file_path, 'w', encoding='utf-8') as file:
    file.write(lowercase_text)

print("Lowercasing completed and saved to the same file.")
