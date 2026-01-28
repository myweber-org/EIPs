
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str, optional): Path to save the cleaned CSV file.
                                     If None, overwrites the input file.
        subset (list, optional): List of column names to consider for duplicates.
    
    Returns:
        int: Number of duplicate rows removed.
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        
        if subset:
            df_cleaned = df.drop_duplicates(subset=subset, keep='first')
        else:
            df_cleaned = df.drop_duplicates(keep='first')
        
        final_rows = len(df_cleaned)
        duplicates_removed = initial_rows - final_rows
        
        if output_file is None:
            output_file = input_file
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Removed {duplicates_removed} duplicate rows.")
        print(f"Original rows: {initial_rows}")
        print(f"Cleaned rows: {final_rows}")
        print(f"Saved to: {output_file}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return -1
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        return -1
    except Exception as e:
        print(f"Error: {str(e)}")
        return -1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)