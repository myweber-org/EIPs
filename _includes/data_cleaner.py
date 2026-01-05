
import pandas as pd

def clean_dataframe(df, subset=None, fillna_strategy='drop', fillna_value=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    subset (list, optional): Column labels to consider for identifying duplicates.
                             If None, all columns are used.
    fillna_strategy (str): Strategy to handle missing values.
                           Options: 'drop', 'fill', 'ffill', 'bfill'.
    fillna_value: Value to use when fillna_strategy is 'fill'.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    cleaned_df = df.copy()

    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates(subset=subset, keep='first')

    # Handle missing values
    if fillna_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fillna_strategy == 'fill':
        if fillna_value is not None:
            cleaned_df = cleaned_df.fillna(fillna_value)
        else:
            raise ValueError("fillna_value must be provided when fillna_strategy is 'fill'")
    elif fillna_strategy == 'ffill':
        cleaned_df = cleaned_df.ffill()
    elif fillna_strategy == 'bfill':
        cleaned_df = cleaned_df.bfill()
    else:
        raise ValueError(f"Unsupported fillna_strategy: {fillna_strategy}")

    return cleaned_df

def main():
    # Example usage
    data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)

    cleaned = clean_dataframe(df, subset=['A', 'C'], fillna_strategy='ffill')
    print("\nCleaned DataFrame:")
    print(cleaned)

if __name__ == "__main__":
    main()