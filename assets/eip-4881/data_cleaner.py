import csv
import sys

def clean_csv(input_file, output_file, delimiter=',', quotechar='"'):
    """
    Clean a CSV file by removing rows with missing values in required columns.
    """
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter=delimiter, quotechar=quotechar)
            headers = next(reader)
            
            required_columns = [0, 1, 2]  # Example: first three columns are required
            
            cleaned_rows = [headers]
            
            for row_num, row in enumerate(reader, start=2):
                if len(row) >= max(required_columns) + 1:
                    missing = False
                    for col_index in required_columns:
                        if not row[col_index] or row[col_index].strip() == '':
                            print(f"Warning: Missing value at row {row_num}, column {col_index}")
                            missing = True
                            break
                    if not missing:
                        cleaned_rows.append(row)
                else:
                    print(f"Warning: Row {row_num} has insufficient columns")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, delimiter=delimiter, quotechar=quotechar, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(cleaned_rows)
        
        print(f"Cleaning complete. Cleaned data saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = clean_csv(input_file, output_file)
    sys.exit(0 if success else 1)