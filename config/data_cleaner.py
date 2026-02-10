import pandas as pd

def clean_dataset(df, subset_columns=None, fill_strategy='mean'):
    """
    Cleans a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset_columns (list, optional): Columns to consider for duplicate removal.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Remove duplicates
    if subset_columns:
        df_clean = df_clean.drop_duplicates(subset=subset_columns)
    else:
        df_clean = df_clean.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    categorical_cols = df_clean.select_dtypes(exclude=['number']).columns
    
    if fill_strategy == 'drop':
        df_clean = df_clean.dropna()
    elif fill_strategy == 'mean':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        df_clean[categorical_cols] = df_clean[categorical_cols].fillna(df_clean[categorical_cols].mode().iloc[0])
    elif fill_strategy == 'median':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        df_clean[categorical_cols] = df_clean[categorical_cols].fillna(df_clean[categorical_cols].mode().iloc[0])
    elif fill_strategy == 'mode':
        for col in df_clean.columns:
            if col in numeric_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
    else:
        raise ValueError("Invalid fill_strategy. Choose from 'mean', 'median', 'mode', or 'drop'.")
    
    return df_clean

# Example usage (commented out)
# if __name__ == "__main__":
#     sample_data = {'A': [1, 2, None, 4, 4],
#                    'B': [5, None, 7, 8, 8],
#                    'C': ['x', 'y', 'z', None, 'x']}
#     df = pd.DataFrame(sample_data)
#     cleaned_df = clean_dataset(df, fill_strategy='mean')
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)