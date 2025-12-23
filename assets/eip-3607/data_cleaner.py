
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values.")
            
            # Handle missing values based on strategy
            if missing_strategy == 'mean':
                df = df.fillna(df.mean(numeric_only=True))
            elif missing_strategy == 'median':
                df = df.fillna(df.median(numeric_only=True))
            elif missing_strategy == 'mode':
                df = df.fillna(df.mode().iloc[0])
            elif missing_strategy == 'drop':
                df = df.dropna()
            else:
                df = df.fillna(0)
                
            print(f"Missing values handled using '{missing_strategy}' strategy.")
        
        # Remove duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            print(f"Removed {duplicates} duplicate rows.")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Final data shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

def validate_data(df, required_columns=None):
    """
    Validate data structure and content.
    """
    if df is None or df.empty:
        print("Error: Data is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    # Check for negative values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f"Warning: Column '{col}' contains negative values.")
    
    return True

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_csv, output_csv, missing_strategy='median')
    
    if cleaned_df is not None:
        is_valid = validate_data(cleaned_df)
        if is_valid:
            print("Data cleaning completed successfully.")
        else:
            print("Data validation failed.")