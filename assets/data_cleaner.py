import re
import unicodedata

def normalize_text(text):
    """Normalize text by removing extra whitespace and converting to lowercase."""
    if not isinstance(text, str):
        return ''
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def remove_special_characters(text, keep_spaces=True):
    """Remove special characters from text."""
    if not isinstance(text, str):
        return ''
    if keep_spaces:
        pattern = r'[^a-zA-Z0-9\s]'
    else:
        pattern = r'[^a-zA-Z0-9]'
    return re.sub(pattern, '', text)

def clean_whitespace(text):
    """Clean excessive whitespace characters."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def normalize_unicode(text, form='NFKD'):
    """Normalize unicode characters to their ASCII representation."""
    if not isinstance(text, str):
        return ''
    normalized = unicodedata.normalize(form, text)
    return ''.join([c for c in normalized if not unicodedata.combining(c)])

def full_clean(text, normalize_unicode_flag=True):
    """Apply all cleaning functions to text."""
    if normalize_unicode_flag:
        text = normalize_unicode(text)
    text = normalize_text(text)
    text = remove_special_characters(text)
    text = clean_whitespace(text)
    return text

if __name__ == '__main__':
    sample_text = "  Hello   World!!  \nThis is a test.  "
    print(f"Original: '{sample_text}'")
    print(f"Cleaned: '{full_clean(sample_text)}'")