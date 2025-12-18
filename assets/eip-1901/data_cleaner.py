import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: Input DataFrame
        drop_duplicates: Whether to drop duplicate rows
        fill_missing: Whether to fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate that the DataFrame meets certain criteria.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
    
    Returns:
        Boolean indicating if validation passed
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

def process_data(file_path, output_path=None):
    """
    Main function to process and clean data from a CSV file.
    
    Args:
        file_path: Path to input CSV file
        output_path: Path to save cleaned data (optional)
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        
        if not validate_data(df):
            return None
        
        cleaned_df = clean_dataset(df)
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return cleaned_df
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    result = process_data(input_file, output_file)
    
    if result is not None:
        print(f"Data cleaning completed. Shape: {result.shape}")