import sys
import os
import openai
from llama_index.legacy import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.legacy.indices.managed.colbert_index import ColbertIndex
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.llms.openai import OpenAI as OpenAiLlm
from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate
import logging

def main():
    # Retrieve OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("API key for OpenAI is not set in the environment variables.")

    # Set OpenAI API key
    openai.api_key = api_key

    # Check if storage already exists
    PERSIST_DIR = "./storage_bert"
    if not os.path.exists(PERSIST_DIR):
        print("Hii")
        documents = SimpleDirectoryReader("/Users/samveg.shah/Desktop/Llama-index/data2").load_data()
        print(documents)
        index = ColbertIndex.from_documents(documents)
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
            "'The context information ...' or anything along rather use terms like 'based on the policy documents'  \n"
            "3. Use phrases like 'this may be the section that references your question' instead of 'this will be covered\n"
            "4. Use words like, may and perhaps if needed where information is not clear \n"
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

    # Initialize LLM
    llm = OpenAiLlm(model="gpt-4", max_tokens=1024)

    # Chat messages
    messages = [
        ChatMessage(
            role="system", content="You will be provided with an email/thread of emails and your responsibility is to extract the portion which has the user questions. The format should be assistant:....."
        ),
        ChatMessage(role="user", content="""Hey Kelly,

        Does Funds Transfer cover 3rd party funds such as escrow?
        ...
        amwins.com.""")
    ]

    resp = str(llm.chat(messages)).split(":")[1]
    print("Extracted Question:", resp)

    # Example query
    query = """
    Hello, do you have any cryptojacking included in your policy? 
    """
    
    response = index.as_query_engine(
        text_qa_template=CHAT_TEXT_QA_PROMPT,
        llm=llm,
        similarity_top_k=5
    ).query(query)

    # Print the response
    print("Response:", response)

    # Extract the source passages from the response
    source_passages = []
    for node in response.source_nodes:
        if hasattr(node, 'text'):
            passage_text = node.text
        else:
            passage_text = display_source_node(node)
        if passage_text:
            source_passages.append(passage_text)
            print("\n===")
            print(passage_text)

if __name__ == '__main__':
    main()


# from colbert import Indexer, Searcher

# print(Indexer)
# print(Searcher)