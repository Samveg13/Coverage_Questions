import os
import openai
import pandas as pd
import sys
import logging
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
    PromptTemplate
)
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.llms.openai import OpenAI as OpenAiLlm
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Retrieve OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("API key for OpenAI is not set in the environment variables.")

# Set OpenAI API key
openai.api_key = api_key

# Check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Define the system and user prompts
TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information which are from policy documents, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Directly reference the given context in your answer wherever necessary.\n"
        "2. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along rather use terms like 'based on the policy documents'\n"
        "3. Use phrases like 'this may be the section that references your question' instead of 'this will be covered'\n"
        "4. Use words like, may and perhaps if needed where information is not clear\n"
        "5. Don’t use words that indicate a definitive coverage to avoid liability"
    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

# Load the Large Language Model from OpenAI
llm = OpenAiLlm(model="gpt-4", max_tokens=1024)  # Ensure the correct model name

# Function to process a column from the Excel file
def process_row(description, index2):
    # print(description)
    PERSIST_DIR = "./storage"
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    
    print(f"Processing row {index2}...")
    
    messages = [
        ChatMessage(
            role="system",
            content="You will be provided with an email/thread of emails and your responsibility is to extract the portion which has the user questions. The format should be assistant:....."
        ),
        ChatMessage(role="user", content=description)
    ]
    
    resp = str(llm.chat(messages)).split(":")[1].strip()
    print(f"Extracted User Query ({index2}):", resp)
    
    response = index.as_query_engine(
        text_qa_template=CHAT_TEXT_QA_PROMPT,
        llm=llm,
    ).query(resp)
    
    # Print the response
    print(f"Response ({index2}):", response)
    
    # Extract the source passages from the response
    source_passages = []
    for node in response.source_nodes:
        if hasattr(node, 'text'):
            passage_text = node.text
        else:
            passage_text = display_source_node(node)
        if passage_text:
            source_passages.append(passage_text)
    
    # Append results to results_df
    return resp, response, source_passages

# Load Excel file
excel_file = '/Users/samveg.shah/Desktop/Llama-index/strict.csv'
df = pd.read_csv(excel_file)

# Ensure the DESCRIPTION column exists
if 'DESCRIPTION' not in df.columns:
    raise ValueError("The Excel file does not contain a 'DESCRIPTION' column.")

# Add new columns for responses
df['Extracted User Query'] = ""
df['Response'] = ""
df['Source Passages'] = ""

# Process each row in the Excel file
for index, row in df.iterrows():
    description = row['DESCRIPTION']
    extracted_query, response, source_passages = process_row(description, index)
    
    df.at[index, 'Extracted User Query'] = extracted_query
    df.at[index, 'Response'] = response
    df.at[index, 'Source Passages'] = "\n".join(source_passages)

    print(f"Completed processing row {index}.")

# Save the updated DataFrame to a new Excel file
df.to_excel('processed_responses.xlsx', index=False)

print("Processing complete.")