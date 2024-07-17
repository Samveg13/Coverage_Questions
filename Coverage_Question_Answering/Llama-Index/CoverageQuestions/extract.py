import pandas as pd

# Path to your input CSV file
input_file_path = "/Users/samveg.shah/Desktop/experimentation/new_fin_strict.csv"

# Path to save the new filtered CSV file
output_file_path = "/Users/samveg.shah/Desktop/Llama-index/output_file_only_doc2.csv"

# Read the CSV file
df = pd.read_csv(input_file_path)

# Filter the rows where the 'answer' column contains the phrase 'Can be answered'
filtered_df = df[df['answer2'].str.contains('Can be answered', na=False)]

# Write the filtered data to a new CSV file
filtered_df.to_csv(output_file_path, index=False)

print(f"Filtered data has been written to {output_file_path}")