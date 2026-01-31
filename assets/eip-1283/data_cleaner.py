import pandas as pd
import argparse
import sys

def remove_duplicates(input_file, output_file, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        subset (list, optional): Columns to consider for duplicates
        keep (str): Which duplicate to keep - 'first', 'last', or False to drop all
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        final_count = len(df_cleaned)
        
        df_cleaned.to_csv(output_file, index=False)
        
        duplicates_removed = initial_count - final_count
        print(f"Successfully processed {input_file}")
        print(f"Initial rows: {initial_count}")
        print(f"Final rows: {final_count}")
        print(f"Duplicates removed: {duplicates_removed}")
        print(f"Cleaned data saved to: {output_file}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Remove duplicate rows from CSV files')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--subset', nargs='+', help='Columns to consider for duplicates')
    parser.add_argument('--keep', choices=['first', 'last', 'none'], 
                       default='first', help='Which duplicate to keep')
    
    args = parser.parse_args()
    
    keep_value = False if args.keep == 'none' else args.keep
    remove_duplicates(args.input, args.output, args.subset, keep_value)

if __name__ == '__main__':
    main()