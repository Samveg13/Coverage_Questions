import pandas as pd

def clean_text(text):
    # Remove all types of whitespace and quotes, including inner spaces
    return text.replace('\n','').replace('\r','').replace('\t','').replace('"','').replace(' ','').strip()

def read_and_compare_columns(file1_path, file2_path, column1_name, column2_name, output_excel_path):
    # Read the CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # Check if the specified columns exist in the CSV files
    if column1_name not in df1.columns:
        raise ValueError(f"The column '{column1_name}' does not exist in the CSV file '{file1_path}'.")
    if column2_name not in df2.columns:
        raise ValueError(f"The column '{column2_name}' does not exist in the CSV file '{file2_path}'.")
    
    # Save original texts
    original_col1 = df1[column1_name]
    original_col2 = df2[column2_name]
    
    # Clean text in the specified columns for comparison
    col1 = original_col1.astype(str).apply(clean_text)
    col2 = original_col2.astype(str).apply(clean_text)
    
    # Ensure both columns have the same length
    if len(col1) != len(col2):
        raise ValueError("The columns have different lengths and cannot be compared row by row.")
    
    # Compare columns and calculate if Extracted User Query (col2) is a substring of openai_response (col1)
    matches = [query in response for query, response in zip(col2, col1)]
    match_count = sum(matches)
    total_count = len(col1)
    
    # Identify non-matching values based on cleaned text
    non_matches = ~pd.Series(matches)
    non_matching_df = pd.DataFrame({
        f"{column1_name}_file1": original_col1[non_matches].reset_index(drop=True),
        f"{column2_name}_file2": original_col2[non_matches].reset_index(drop=True)
    })
    
    # Save non-matching values to a CSV file
    non_matching_df.to_csv(output_excel_path, index=False)
    
    # Print matching statistics
    match_percentage = (match_count / total_count) * 100
    print(f"Total values: {total_count}")
    print(f"Matching values: {match_count}")
    print(f"Percentage of matching values: {match_percentage:.2f}%")
    print(f"Non-matching values saved to: {output_excel_path}")

if __name__ == "__main__":
    # Define file paths, column names, and output CSV path
    file1_path = "/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Generated/updated_file.csv"  # Replace with your first CSV file path
    file2_path = "/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Dataset/data_finale.csv"  # Replace with your second CSV file path
    column1_name = "openai_response"  # Replace with the name of the column in the first CSV file
    column2_name = "Extracted User Query"  # Replace with the name of the column in the second CSV file
    op = "/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Generated/compare.csv"  # Output CSV file path
    read_and_compare_columns(file1_path, file2_path, column1_name, column2_name, op)