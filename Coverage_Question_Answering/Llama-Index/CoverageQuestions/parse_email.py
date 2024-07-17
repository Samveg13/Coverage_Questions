import openai
import pandas as pd

def initialize_openai(api_key):
    openai.api_key = api_key

def get_chat_response(prompt, model="gpt-4", temperature=0.7, max_tokens=1024):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_csv_and_get_responses(api_key, csv_file_path):
    initialize_openai(api_key)
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Check if the 'description' column exists
    if 'DESCRIPTION' not in df.columns:
        print("Error: 'description' column not found in the CSV file.")
        return
    
    responses = []
    
    for index, row in df.iterrows():
        email_content = row['DESCRIPTION']
        prompt = f"""This is an email or thread of email between the cyber insurance provider named Coalition and the client. In the following email, the client has asked a query, question, or seeks clarification on something. Can you extract the part where that is? Make sure to extract the entire query and not just some lines of the query; the query might be spaced out on a number of lines. The email is: "{email_content}" """
        
        response = get_chat_response(prompt)
        responses.append(response)
        print(f"Response for row {index}:", response)
    
    # Add the responses to a new column 'parsed_email'
    df['parsed_email'] = responses
    
    # Save the DataFrame back to the CSV file
    df.to_csv(csv_file_path, index=False)

# Example Usage
if __name__ == "__main__":
    api_key = "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"
    csv_file_path = "/Users/samveg.shah/Desktop/Llama-index/processed_responses - Sheet1.csv"
    process_csv_and_get_responses(api_key, csv_file_path)