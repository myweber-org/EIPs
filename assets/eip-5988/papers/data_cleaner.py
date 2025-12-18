import csv
import sys

def clean_csv(input_file, output_file):
    """
    Clean a CSV file by removing rows with missing values
    and trimming whitespace from all string fields.
    """
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            
            cleaned_rows = []
            for row in reader:
                # Skip rows with missing values
                if any(field.strip() == '' for field in row):
                    continue
                
                # Trim whitespace from all fields
                cleaned_row = [field.strip() for field in row]
                cleaned_rows.append(cleaned_row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            writer.writerows(cleaned_rows)
        
        print(f"Cleaned data saved to {output_file}")
        print(f"Removed {len(cleaned_rows)} valid rows out of {len(cleaned_rows) + (reader.line_num - 1 - len(cleaned_rows))} total rows")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_csv(input_file, output_file)