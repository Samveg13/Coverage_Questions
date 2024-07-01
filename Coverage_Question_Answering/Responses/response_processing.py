import os
import openai

from openai import OpenAI
import pandas as pd
from prompts import SYSTEM_PROMPT, generate_prompt
import re

# Function to setup OpenAI API key
def setup_openai_api_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError('API key for OpenAI is not set in the environment variables.')
    openai.api_key = api_key

def extract_content(response_str):
    match = re.search(r"content='(.*?)', role=", response_str)
    if match:
        content = match.group(1)
        return content.replace("\\'", "'")
    return None

# Function to process each row
def process_row(prompt, model="gpt-4o", temperature=0.7, max_tokens=1024):
    client = OpenAI()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to process a batch
def process_batch(batch_df):
    for index, row in batch_df.iterrows():
        query = row['Extracted User Query']
        responses = row['Complete Responses']
        prompt = generate_prompt(question=query, email_content=responses)
        openai_response = process_row(prompt)
        batch_df.at[index, 'extracted_responses'] = openai_response
        print(f"Processed row {index} - OpenAI Response: {openai_response}")

# Main function to read CSV, process entries, and save updated CSV
def main():
    setup_openai_api_key()
    
    # Read the CSV file
    df = pd.read_csv('/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Dataset/dataset_coverage_fin.csv')

    # Check if required columns are present
    if 'DESCRIPTION' not in df.columns:
        raise ValueError("The CSV file must contain 'DESCRIPTION' column.")

    # Add 'extracted_responses' column if it doesn't exist
    if 'extracted_responses' not in df.columns:
        df['extracted_responses'] = ''

    # Define batch size
    batch_size = 20
    
    # Iterate over the DataFrame in batches of 20
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch_df = df.iloc[start:end]
        process_batch(batch_df)
        
        # After processing the batch, update the original DataFrame with results
        df.update(batch_df)

    # Save the updated dataframe back to the same CSV file
    df.to_csv('/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Dataset/dataset_coverage_fin.csv', index=False)

if __name__ == "__main__":
    main()