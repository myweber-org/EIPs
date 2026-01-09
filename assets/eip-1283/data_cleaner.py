import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and optionally save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")

        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")

        # Drop columns with all missing values
        df = df.dropna(axis=1, how='all')
        print(f"After dropping empty columns: {df.shape}")

        # Fill numeric column missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Fill categorical column missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)

        # Remove leading/trailing whitespace from string columns
        for col in categorical_cols:
            df[col] = df[col].str.strip()

        print(f"Final cleaned shape: {df.shape}")
        print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")

        return df

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate the structure and content of a DataFrame.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None.")
        return False

    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False

    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            print(f"Column {col} contains infinite values.")
            return False

    return True

if __name__ == "__main__":
    # Example usage
    cleaned_df = clean_csv_data("sample_data.csv", "cleaned_data.csv")
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        print(f"Data validation result: {is_valid}")