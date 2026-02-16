
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_dataimport pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    remove_duplicates (bool): Whether to remove duplicate rows
    fill_missing (str or dict): Strategy for filling missing values:
        - 'mean': Fill with column mean (numeric only)
        - 'median': Fill with column median (numeric only)
        - 'mode': Fill with column mode
        - dict: Column-specific fill values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing is not None:
        if cleaned_df.isnull().sum().sum() > 0:
            if isinstance(fill_missing, dict):
                cleaned_df = cleaned_df.fillna(fill_missing)
            elif fill_missing == 'mean':
                cleaned_df = cleaned_df.fillna(cleaned_df.select_dtypes(include=['number']).mean())
            elif fill_missing == 'median':
                cleaned_df = cleaned_df.fillna(cleaned_df.select_dtypes(include=['number']).median())
            elif fill_missing == 'mode':
                for col in cleaned_df.columns:
                    if cleaned_df[col].dtype == 'object':
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0)
            print(f"Filled missing values using {fill_missing} strategy")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve'],
        'score': [85, 92, 92, 78, None],
        'department': ['HR', 'IT', 'IT', 'Finance', 'HR']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    cleaned = clean_dataset(df, remove_duplicates=True, fill_missing='mode')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'name', 'score'])
    print(f"\nDataFrame validation: {is_valid}")