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
        df.drop_duplicates(inplace=True)
        print(f"After removing duplicates: {df.shape}")

        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)

        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)

        # Remove columns with more than 50% missing values
        threshold = len(df) * 0.5
        df.dropna(thresh=threshold, axis=1, inplace=True)
        print(f"After dropping high-missing columns: {df.shape}")

        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")

        return df

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def validate_dataframe(df):
    """
    Perform basic validation on a DataFrame.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False

    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }

    print("Data Validation Results:")
    for key, value in validation_results.items():
        print(f"  {key}: {value}")

    return validation_results['missing_values'] == 0 and validation_results['duplicate_rows'] == 0

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"

    cleaned_df = clean_csv_data(input_file, output_file)

    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        print(f"Data validation passed: {is_valid}")