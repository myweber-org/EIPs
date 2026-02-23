import re
import unicodedata

def clean_text(text, remove_digits=False, remove_punctuation=False, to_lower=True):
    """
    Clean and normalize a given text string.

    Args:
        text (str): Input text to be cleaned.
        remove_digits (bool): If True, remove all digits from the text.
        remove_punctuation (bool): If True, remove all punctuation characters.
        to_lower (bool): If True, convert text to lowercase.

    Returns:
        str: Cleaned and normalized text.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Normalize unicode characters (e.g., convert accented characters to their base form)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Optionally convert to lowercase
    if to_lower:
        text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Optionally remove digits
    if remove_digits:
        text = re.sub(r'\d+', '', text)

    # Optionally remove punctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)

    return text

def tokenize_text(text, tokenizer=None):
    """
    Tokenize the cleaned text. If no custom tokenizer is provided, split by whitespace.

    Args:
        text (str): Cleaned text to tokenize.
        tokenizer (callable, optional): A function that takes a string and returns a list of tokens.
                                        Defaults to splitting by whitespace.

    Returns:
        list: List of tokens.
    """
    if tokenizer is None:
        tokenizer = lambda x: x.split()
    return tokenizer(text)

if __name__ == "__main__":
    # Example usage
    sample_text = "Hello, World! This is a sample text with numbers 123 and punctuation!!!"
    cleaned = clean_text(sample_text, remove_digits=True, remove_punctuation=True)
    tokens = tokenize_text(cleaned)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
    print(f"Tokens: {tokens}")