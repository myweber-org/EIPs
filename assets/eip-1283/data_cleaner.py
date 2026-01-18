import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str or dict): Method to fill missing values. 
                                Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
                                If None, missing values are not filled.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
    
    return cleaned_df

def validate_dataset(df, required_columns=None, unique_columns=None):
    """
    Validate a DataFrame for required columns and unique constraints.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    unique_columns (list): List of column names that should have unique values.
    
    Returns:
    dict: Dictionary with validation results and any error messages.
    """
    validation_result = {
        'is_valid': True,
        'errors': []
    }
    
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_columns}")
    
    if unique_columns is not None:
        for col in unique_columns:
            if col in df.columns:
                if df[col].duplicated().any():
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(f"Column '{col}' contains duplicate values")
    
    return validation_result

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve'],
        'age': [25, 30, 30, 35, None],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    validation = validate_dataset(cleaned, required_columns=['id', 'name'], unique_columns=['id'])
    print("Validation Result:")
    print(validation)