
import csv
import hashlib
from collections import defaultdict

def generate_row_hash(row):
    """Generate a hash for a row to identify duplicates."""
    row_string = ''.join(str(value) for value in row)
    return hashlib.md5(row_string.encode()).hexdigest()

def remove_duplicates(input_file, output_file):
    seen_hashes = defaultdict(int)
    unique_rows = []

    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        for row in reader:
            row_hash = generate_row_hash(row)
            if seen_hashes[row_hash] == 0:
                seen_hashes[row_hash] += 1
                unique_rows.append(row)

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(unique_rows)

    print(f"Original rows: {len(seen_hashes) + len(unique_rows) - sum(seen_hashes.values())}")
    print(f"Unique rows written: {len(unique_rows)}")

if __name__ == "__main__":
    remove_duplicates('input_data.csv', 'cleaned_data.csv')