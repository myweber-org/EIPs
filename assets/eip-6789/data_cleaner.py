
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]import pandas as pd
import numpy as np

def clean_csv_data(file_path, strategy='mean', columns=None):
    """
    Clean a CSV file by handling missing values in specified columns.
    
    Args:
        file_path (str): Path to the CSV file.
        strategy (str): Strategy for handling missing values. 
                       Options: 'mean', 'median', 'mode', 'drop'.
        columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    else:
        columns_to_clean = [col for col in columns if col in df.columns]
    
    for col in columns_to_clean:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            elif strategy == 'drop':
                df = df.dropna(subset=[col])
                continue
            else:
                print(f"Warning: Unknown strategy '{strategy}'. Using 'mean'.")
                fill_value = df[col].mean()
            
            df[col].fillna(fill_value, inplace=True)
            print(f"Filled missing values in column '{col}' using {strategy} strategy.")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a new CSV file.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame.
        output_path (str): Path for the output CSV file.
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to '{output_path}'.")
    else:
        print("No data to save.")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, strategy='median')
    save_cleaned_data(cleaned_df, output_file)