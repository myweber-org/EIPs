
import re
import unicodedata

def clean_text(text, remove_digits=False, keep_case=False):
    """
    Clean and normalize input text by removing extra whitespace,
    optionally removing digits, and optionally preserving case.
    """
    if not isinstance(text, str):
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if remove_digits:
        text = re.sub(r'\d+', '', text)

    if not keep_case:
        text = text.lower()

    return text

def tokenize_text(text, pattern=r'\w+'):
    """
    Tokenize text using a regex pattern.
    Default pattern matches alphanumeric words.
    """
    tokens = re.findall(pattern, text)
    return tokens

def remove_stopwords(tokens, stopwords=None):
    """
    Remove stopwords from a list of tokens.
    If no stopwords provided, uses a minimal default set.
    """
    if stopwords is None:
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at'}

    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    return filtered_tokens