import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    if drop_duplicates:
        df = df.drop_duplicates()
    
    if fill_missing:
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def process_data_file(input_path, output_path, required_columns=None):
    """
    Process a data file: read, clean, validate, and save.
    """
    try:
        df = pd.read_csv(input_path)
        
        df = clean_dataframe(df)
        
        validate_dataframe(df, required_columns)
        
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        
        return df
        
    except Exception as e:
        print(f"Error processing file: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    required_cols = ["id", "name", "value"]
    
    try:
        cleaned_df = process_data_file(input_file, output_file, required_cols)
        print(f"Data cleaning completed. Shape: {cleaned_df.shape}")
    except Exception as e:
        print(f"Failed to process data: {e}")