import openai
import pandas as pd
from ragatouille import RAGPretrainedModel
from openai import OpenAI
import os
import re

def initialize_openai(api_key):
    openai.api_key = api_key

def extract_content(response_str):
    # Regular expression to extract the content inside the single quotes following "content="
    match = re.search(r"content='(.*?)', role=", response_str)
    if match:
        content = match.group(1)
        # Unescape single quotes inside the content
        return content.replace("\\'", "'")
    else:
        return None

def get_chat_response(prompt, model="gpt-4o", temperature=0.7, max_tokens=1024):
    try:
        client = OpenAI()

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are an expert Q&A system that is trusted around the world.\n
                    Always answer the query using the provided context information which are from policy documents, 
                    and not prior knowledge.\n
                    Some rules to follow:\n
                    1. Directly reference the given context in your answer wherever necessary.\n
                    2. Avoid statements like 'Based on the context, ...' or 
                    'The context information ...' or anything along rather use terms like 'based on the policy documents'  \n
                    3. Use phrases like “this may be the section that references your question” instead of “this will be covered\n
                    4. Use words like, may and perhaps if needed where information is not clear \n
                    5. Don’t use words that indicate a definitive coverage to avoid liability

                    Please note: Some queries may be in form of a situation, use the context provided to answer the questions which follows the situation and justify  """},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,

        )

        return extract_content(str(completion.choices[0].message))
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_excel_and_get_responses(api_key, file_path, output_file_path):
    initialize_openai(api_key)
    
    # Read the Excel file
    df = pd.read_csv(file_path)
    
    # Check if the 'Extracted User Query' column exists
    if 'Extracted User Query' not in df.columns:
        print("Error: 'Extracted User Query' column not found in the Excel file.")
        return
    
    # Initialize RAGPretrainedModel
    RAG = RAGPretrainedModel.from_index("/Users/samveg.shah/Desktop/Llama-index/.ragatouille/colbert/indexes/my_index")
    
    responses = []
    
    for index, row in df.iterrows():
        email_content = row['Extracted User Query']
        
        # Check if the cell in 'Extracted User Query' is empty
        if not email_content or pd.isna(email_content):
            print(f"Warning: Empty cell found in row {index} of the 'Extracted User Query' column. Skipping this row.")
            responses.append(None)
            break
        
        # Get context using RAG model
        results = RAG.search(query=email_content, k=10)
        context = "\n".join([x['content'] for x in results])

        # Prepare prompt
        prompt = f"""
        The email provided is : {email_content}
        The context is: {context}
        """
        print(email_content)
        
        # Get OpenAI response
        response = get_chat_response(prompt)
        responses.append(response)
        print(f"Response for row {index}:", response)
    
    # Add the responses to a new column 'colbert_response'
    df['colbert_response'] = responses
    
    # Save the DataFrame back to the Excel file
    df.to_csv(output_file_path, index=False)

# Example Usage
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    # api_key = "YOUR_OPENAI_API_KEY"
    file_path = "/Users/samveg.shah/Desktop/Llama-index/processed_responses_fin2.0 - processed_responses - Sheet1.csv"
    output_file_path = "/Users/samveg.shah/Desktop/Llama-index/col_fin.csv"
    process_excel_and_get_responses(api_key, file_path, output_file_path)