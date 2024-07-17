import os
import openai
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    PropertyGraphIndex
)
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.llms.openai import OpenAI as OpenAiLlm
from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.legacy.retrievers import BM25Retriever
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import AutoMergingRetriever

# Retrieve OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("API key for OpenAI is not set in the environment variables.")

# Set OpenAI API key
openai.api_key = api_key

# Check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("/Users/samveg.shah/Desktop/Llama-index/data").load_data()
    
    # Create the pipeline with transformations
    # pipeline = IngestionPipeline(
    #     transformations=[
    #         SentenceSplitter(chunk_size=512, chunk_overlap=0),
    #         TitleExtractor(),
    #         OpenAIEmbedding(),
    #     ],
    #     documents=documents,  # Provide documents directly to the pipeline
        
    # )

    # # Run the pipeline
    # nodes = pipeline.run()
    
    # Create an index from documents
    index = VectorStoreIndex.from_documents(documents=documents,transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=0),
            TitleExtractor(),
            OpenAIEmbedding(),
        ],)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Either way we can now query the index

TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information which are from policy documents, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Directly reference the given context in your answer wherever necessary.\n"
        "2. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along rather use terms like 'based on the policy documents'  \n"
        "3. Use phrases like “this may be the section that references your question” instead of “this will be covered\n"
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

# Initialize the LLM
llm = OpenAiLlm(model="gpt-4", max_tokens=1024)  # Ensure the correct model name

# Define the messages
messages = [
    ChatMessage(
        role="system", content="You will be provided with an email/thread of emails and your responsibility is to extract the portion which has the user questions. The format should be assistant:....."
    ),
    ChatMessage(role="user", content="""Hey Kelly,

    Does Funds Transfer cover 3rd party funds such as escrow? 

    From: Kelly Brugler <kelly@coalitioninc.com>
    Sent: Monday, January 22, 2024 11:28 AM
    To: Jesse Lindemulder <jesse.lindemulder@amwins.com>
    Cc: Amber Artis <amber@coalitioninc.com>
    Subject: CONTINGENT Quote for Starfield & Smith, P.C. (Limit: $5M, Retention: $25K, Premium: $24,987)

    Hi Jesse Lindemulder,

    Thank you for requesting a quote for Starfield & Smith, P.C.. We’ve completed the review and released this quote subject to a contingency.
    ...""")
]

# Generate the response from the LLM
llm_response = llm.chat(messages)
extracted_question = str(llm_response).split(":")[1].strip()
extracted_question="Hello, do you have any cryptojacking included in your policy?"
print("Extracted question:", extracted_question)

# Query the index
response = index.as_query_engine(
    text_qa_template=CHAT_TEXT_QA_PROMPT,
    llm=llm,
    similarity_top_k=10
).query(extracted_question)

# Print the response
print("Response:", response)

# Now extract the source passages from the response
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
base_retriever = index.as_retriever(similarity_top_k=5)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)
# query_str = "What were some lessons learned from red-teaming?"
# query_str = "Can you tell me about the key concepts for safety finetuning"
query_str = (
    "Samveg Shah"
)

nodes = retriever.retrieve(query_str)
base_nodes = base_retriever.retrieve(query_str)
z=0
for node in nodes:
    if hasattr(node, 'text'):
        print("\n===\n")
        print(z)
        z=z+1
        print(node.text)
    else:
        print(node)
