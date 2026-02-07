import csv
import sys

def clean_csv(input_file, output_file):
    """
    Clean CSV file by removing rows where all fields are empty or whitespace,
    and trimming leading/trailing whitespace from all fields.
    """
    cleaned_rows = []
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for row in reader:
                # Trim whitespace from each field
                trimmed_row = [field.strip() for field in row]
                # Skip rows where all fields are empty after trimming
                if any(field != '' for field in trimmed_row):
                    cleaned_rows.append(trimmed_row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(cleaned_rows)
            
        print(f"Cleaned data saved to {output_file}")
        print(f"Original rows: {len(list(csv.reader(open(input_file, 'r'))))}")
        print(f"Cleaned rows: {len(cleaned_rows)}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_csv(input_file, output_file)