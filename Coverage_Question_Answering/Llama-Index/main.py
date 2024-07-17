import os
import openai
import pandas as pd
import sys
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.llms.openai import OpenAI as OpenAiLlm
from prompts import CHAT_TEXT_QA_PROMPT

# Initialize OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("API key for OpenAI is not set in the environment variables.")
openai.api_key = api_key

# Global Variables
PERSIST_DIR = "./storage"
llm = OpenAiLlm(model="gpt-4o", max_tokens=1024)  # Ensure the correct model name

# Ensures the index is loaded or created
def ensure_index():
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index

# Process a single row and return the response and source passages
def process_row(description, idx, index):
    response = index.as_query_engine(
        text_qa_template=CHAT_TEXT_QA_PROMPT,
        llm=llm,
        similarity_top_k=10
    ).query(description)

    source_passages = [
        (node.text if hasattr(node, 'text') else display_source_node(node))
        for node in response.source_nodes if node
    ]

    return response, source_passages

def main():
    index = ensure_index()
    # Load Excel file
    excel_file = '/Users/samveg.shah/Desktop/Llama-index/fin_fin.csv'
    df = pd.read_csv(excel_file)[0:108]

    if 'Extracted User Query' not in df.columns:
        raise ValueError("The Excel file does not contain a 'Extracted User Query' column.")
    
    df['Response'] = ""
    df['Source_Passages_AI'] = ""
    
    for idx, row in df.iterrows():
        description = row['Extracted User Query']
        response, source_passages = process_row(description, idx, index)
        
        df.at[idx, 'AI_Response'] = response
        df.at[idx, 'Source_Passages_AI'] = source_passages
        print(f"Completed processing row {idx}.")
    
    df.to_csv('processed_responses_2.csv', index=False, escapechar="\\")
    print("Processing complete.")

if __name__ == "__main__":
    main()