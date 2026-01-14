import pandas as pd

def clean_data(df, column_to_check):
    """
    Clean the input DataFrame by removing duplicates and filling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # Fill missing values in the specified column with the column's mean
    if column_to_check in df_cleaned.columns:
        df_cleaned[column_to_check].fillna(df_cleaned[column_to_check].mean(), inplace=True)

    # Drop rows where any remaining columns have NaN values
    df_cleaned.dropna(inplace=True)

    return df_cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': [10, 11, 12, 12, 14]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    cleaned_df = clean_data(df, 'B')
    print("\nCleaned DataFrame:")
    print(cleaned_df)