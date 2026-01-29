
import pandas as pd
import re

def clean_dataframe(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing
    specified string columns (strip whitespace, lowercase).
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize specified string columns
    for col in column_names:
        if col in df_cleaned.columns:
            # Convert to string, strip whitespace, convert to lowercase
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.lower()
    
    return df_cleaned

def remove_special_characters(text):
    """
    Remove special characters from a string, keeping only alphanumeric and spaces.
    """
    if isinstance(text, str):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def validate_email(email):
    """
    Validate email format using a simple regex pattern.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if isinstance(email, str):
        return bool(re.match(pattern, email))
    return False

def process_data(file_path, output_path, columns_to_clean):
    """
    Main function to load, clean, and save data.
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Clean data
        df_cleaned = clean_dataframe(df, columns_to_clean)
        
        # Apply additional cleaning to specific columns if needed
        if 'email' in df_cleaned.columns:
            df_cleaned['email'] = df_cleaned['email'].apply(remove_special_characters)
        
        # Save cleaned data
        df_cleaned.to_csv(output_path, index=False)
        print(f"Data cleaned and saved to {output_path}")
        
        return df_cleaned
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    columns = ["name", "email", "address"]
    
    cleaned_df = process_data(input_file, output_file, columns)
    if cleaned_df is not None:
        print(f"Cleaned {len(cleaned_df)} records")