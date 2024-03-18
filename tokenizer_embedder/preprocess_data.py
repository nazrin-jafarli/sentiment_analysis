
import re

def preprocess_text(text):
    """
    Preprocesses the text by removing punctuation and converting to lowercase.
    """
    # Remove punctuation using regular expressions
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    # Convert text to lowercase
    text_lower = text_no_punct.lower()
    return text_lower
