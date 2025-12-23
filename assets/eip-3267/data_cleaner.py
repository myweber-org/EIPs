import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate that DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

def process_data_file(file_path, output_path=None):
    """
    Process a data file by loading, cleaning, and optionally saving it.
    
    Args:
        file_path (str): Path to input data file.
        output_path (str): Path to save cleaned data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")
        
        is_valid, message = validate_data(df)
        if not is_valid:
            print(f"Validation warning: {message}")
        
        cleaned_df = clean_dataset(df)
        
        if output_path:
            if output_path.endswith('.csv'):
                cleaned_df.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                cleaned_df.to_excel(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None