import os
import openai
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole

import sys
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.llms.openai import OpenAI as OpenAiLlm
from prompts import CHAT_TEXT_QA_PROMPT
# from llama_index.response.pprint_utils import pprint_response


# Initialize OpenAI API key
def setup_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("API key for OpenAI is not set in the environment variables.")
    openai.api_key = api_key

# Ensures the index is loaded or created
def ensure_index(persist_dir="/Users/samveg.shah/Desktop/Llama-index/storage"):
    if not os.path.exists(persist_dir):
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    return index


# Main function to run the script
def main():
    setup_openai_api_key()
    index = ensure_index()

    llm = OpenAiLlm(model="gpt-4o", max_tokens=1024)  # Ensure the correct model name


    # Placeholder for actual user prompt from user actions or queries
    user_query = """The policy declarations reflect an enhanced waiting period of 8 hours for Item H - Business Interruption and Extra Expenses, but the endorsement (SP14805 11/17) reflects an enhanced waiting period of 1 hour. Which one is correct?"""
    
    response = index.as_query_engine(
            text_qa_template=CHAT_TEXT_QA_PROMPT,
            llm=llm,
            similarity_top_k=10
        ).query(user_query)

    # Print the response
    print("Query Response:", response)

    # Extract and print the source passages
    # source_passages = [
    #     (node.text if hasattr(node, 'text') else display_source_node(node))
    #     for node in response.source_nodes if node
    # ]
    
    # for passage in source_passages:
    #     print("Source Passage:\n", passage)

if __name__ == "__main__":
    main()