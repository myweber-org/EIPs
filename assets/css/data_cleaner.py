
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Cleans a pandas DataFrame by removing duplicates and handling missing values.
    """
    df_clean = df.copy()

    if drop_duplicates:
        initial_rows = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - df_clean.shape[0]
        print(f"Removed {removed} duplicate rows.")

    if fill_missing:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if fill_strategy == 'mean':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif fill_strategy == 'median':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif fill_strategy == 'zero':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
        else:
            raise ValueError("fill_strategy must be 'mean', 'median', or 'zero'")

        non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns
        df_clean[non_numeric_cols] = df_clean[non_numeric_cols].fillna('Unknown')
        print(f"Filled missing values using strategy: {fill_strategy} for numeric, 'Unknown' for non-numeric.")

    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validates the DataFrame for required columns and basic integrity.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise KeyError(f"DataFrame missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("DataFrame is empty.")

    print("DataFrame validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.0, np.nan, 20.0, 25.5],
        'category': ['A', 'B', 'B', None, 'C', 'A']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")

    cleaned_df = clean_dataframe(df, fill_strategy='median')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\n")

    try:
        validate_dataframe(cleaned_df, required_columns=['id', 'value'])
    except Exception as e:
        print(f"Validation error: {e}")