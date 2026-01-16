import pandas as pd

def clean_data(df):
    """
    Remove duplicate rows and fill missing values with column mean.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with column mean
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    
    return df_cleaned

def load_and_clean(file_path):
    """
    Load data from CSV file and apply cleaning.
    """
    try:
        df = pd.read_csv(file_path)
        cleaned_df = clean_data(df)
        return cleaned_df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_data = load_and_clean(input_file)
    
    if cleaned_data is not None:
        cleaned_data.to_csv(output_file, index=False)
        print(f"Cleaned data saved to '{output_file}'")