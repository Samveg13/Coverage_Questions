import os
import re
import openai
import pandas as pd
from ragatouille import RAGPretrainedModel
from openai import OpenAI
from prompts import SYSTEM_PROMPT, generate_prompt

def initialize_openai(api_key):
    openai.api_key = api_key

def extract_content(response_str):
    match = re.search(r"content='(.*?)', role=", response_str)
    if match:
        content = match.group(1)
        return content.replace("\\'", "'")
    return None

def get_chat_response(prompt, model="gpt-4o", temperature=0.7, max_tokens=1024):
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
        return extract_content(str(completion.choices[0].message))
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_excel_and_get_responses(api_key, file_path, output_file_path):
    initialize_openai(api_key)
    
    df = pd.read_csv(file_path)[105:108]
    
    if 'Extracted User Query' not in df.columns:
        print("Error: 'Extracted User Query' column not found in the Excel file.")
        return

    RAG = RAGPretrainedModel.from_index("/Users/samveg.shah/Desktop/Llama-index/.ragatouille/colbert/indexes/my_index")
    
    responses = []
    source_pass = []

    for index, row in df.iterrows():
        email_content = row['Extracted User Query']
        if type(email_content) == float:
            responses.append(None)
            source_pass.append(None)
            continue
        
        results = RAG.search(query=email_content, k=10)
        context = "\n".join([x['content'] for x in results])
        
        prompt = generate_prompt(email_content, context)
        response = get_chat_response(prompt)
        
        responses.append(response)
        source_pass.append(context)

    df['colbert_response'] = responses
    df['col_source_pass'] = source_pass
    
    df.to_csv(output_file_path, index=False, escapechar="\\")

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    file_path = "/Users/samveg.shah/Desktop/Llama-index/Imp/Dataset/data_finale.csv"
    output_file_path = "/Users/samveg.shah/Desktop/Llama-index/Imp/Generated/generation_colbert.csv"
    process_excel_and_get_responses(api_key, file_path, output_file_path)