import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """
    Load a CSV file and perform basic cleaning operations.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Remove duplicate rows
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    removed_duplicates = initial_count - len(df)
    print(f"Removed {removed_duplicates} duplicate rows.")

    # Handle missing values: drop rows where all values are NaN
    df.dropna(how='all', inplace=True)
    # For numeric columns, fill missing values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.median()))

    # Remove outliers using IQR method for numeric columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Cap outliers instead of removing rows
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # Normalize numeric columns to range [0, 1]
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:  # Avoid division by zero
            df[col] = (df[col] - min_val) / (max_val - min_val)

    print(f"Cleaning complete. Final shape: {df.shape}")
    return df

def save_cleaned_data(df, output_filepath):
    """
    Save the cleaned DataFrame to a new CSV file.
    """
    if df is not None:
        try:
            df.to_csv(output_filepath, index=False)
            print(f"Cleaned data saved to {output_filepath}")
        except Exception as e:
            print(f"Error saving file: {e}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"

    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)