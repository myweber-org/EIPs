
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicate rows,
    standardizing column names, and filling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')

    # Fill missing numeric values with column median
    for col in df_cleaned.select_dtypes(include=[np.number]).columns:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # Fill missing categorical values with mode
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        mode_value = df_cleaned[col].mode()
        if not mode_value.empty:
            df_cleaned[col] = df_cleaned[col].fillna(mode_value.iloc[0])
        else:
            df_cleaned[col] = df_cleaned[col].fillna('unknown')

    return df_cleaned

def validate_dataframe(df):
    """
    Perform basic validation on the DataFrame.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    required_columns = ['id', 'name']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if df['id'].duplicated().any():
        raise ValueError("Duplicate IDs found in the DataFrame")

    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'ID': [1, 2, 2, 3, 4],
        'Name': ['Alice', 'Bob', 'Bob', 'Charlie', None],
        'Age': [25, 30, 30, None, 35],
        'Score': [85.5, 92.0, 92.0, 78.5, None]
    }

    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataframe(df)
    print(cleaned_df)

    try:
        validate_dataframe(cleaned_df)
        print("\nData validation passed.")
    except ValueError as e:
        print(f"\nData validation failed: {e}")