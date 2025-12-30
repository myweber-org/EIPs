
import pandas as pd
import re

def clean_dataframe(df, text_column='text', id_column='id'):
    """
    Clean a DataFrame by removing duplicate rows and normalizing text in a specified column.
    """
    # Remove duplicate rows based on the id column
    df_cleaned = df.drop_duplicates(subset=[id_column], keep='first')
    
    # Normalize text: lowercase and remove extra whitespace
    if text_column in df_cleaned.columns:
        df_cleaned[text_column] = df_cleaned[text_column].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).strip().lower())
        )
    
    return df_cleaned

def load_and_clean_csv(file_path, text_column='text', id_column='id'):
    """
    Load a CSV file and clean the data.
    """
    try:
        df = pd.read_csv(file_path)
        cleaned_df = clean_dataframe(df, text_column, id_column)
        return cleaned_df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_data = load_and_clean_csv(input_file)
    
    if cleaned_data is not None:
        cleaned_data.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(cleaned_data)}")