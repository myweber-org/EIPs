
import re

def filter_valid_emails(email_list):
    """
    Filters a list of email addresses, returning only those that match a basic
    email pattern.
    """
    if not isinstance(email_list, list):
        raise TypeError("Input must be a list")
    
    valid_emails = []
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    for email in email_list:
        if isinstance(email, str) and re.match(pattern, email):
            valid_emails.append(email)
    
    return valid_emails