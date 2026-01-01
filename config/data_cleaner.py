
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list (list): Input list that may contain duplicates.
    
    Returns:
        list: List with duplicates removed.
    """
    seen = set()
    result = []
    
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_numeric_data(values, threshold=None):
    """
    Clean numeric data by removing None values and optionally filtering by threshold.
    
    Args:
        values (list): List of numeric values.
        threshold (float, optional): Maximum allowed value. Values above this are removed.
    
    Returns:
        list: Cleaned list of numeric values.
    """
    cleaned = [v for v in values if v is not None]
    
    if threshold is not None:
        cleaned = [v for v in cleaned if v <= threshold]
    
    return cleaned

def validate_email_list(emails):
    """
    Validate and clean a list of email addresses.
    
    Args:
        emails (list): List of email strings.
    
    Returns:
        tuple: (valid_emails, invalid_emails)
    """
    import re
    
    valid = []
    invalid = []
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    for email in emails:
        if isinstance(email, str) and re.match(pattern, email):
            valid.append(email.lower().strip())
        else:
            invalid.append(email)
    
    return valid, invalid

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    emails = ["test@example.com", "invalid-email", "user@domain.org", "another@test.co.uk"]
    valid_emails, invalid_emails = validate_email_list(emails)
    print(f"Valid emails: {valid_emails}")
    print(f"Invalid emails: {invalid_emails}")