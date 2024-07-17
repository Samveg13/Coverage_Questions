import requests
import fitz  # PyMuPDF
import os
import json
import time
import openai
import pandas as pd
from llama_index.core import QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from prompts_2 import SYSTEM_PROMPT, generate_prompt
from openai import OpenAI

from ragatouille import RAGPretrainedModel
from llama_index.core.schema import Document
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings)
from emails import data  # Import the sample_json_obj here
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.llms.openai import OpenAI as OpenAiLlm
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor import LLMRerank

from aws_requests_auth.aws_auth import AWSRequestsAuth
from prompts import CHAT_TEXT_QA_PROMPT
import openai
import pandas as pd
import aiohttp
import asyncio
import time
import os
def initialize_openai(api_key):
    openai.api_key = api_key

async def fetch_response(session, prompt, retry_attempts=5, model="gpt-4o", temperature=0.7, max_tokens=4096):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    backoff_time = 5  # Initial backoff time in seconds
    
    for attempt in range(retry_attempts):
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 429:  # Rate limit error
                    retry_after = int(response.headers.get("Retry-After", backoff_time))
                    print(f"Rate limit reached. Retrying after {retry_after} seconds. Attempt {attempt + 1} of {retry_attempts}.")
                    await asyncio.sleep(retry_after)
                    backoff_time = min(backoff_time * 2, 60)  # Exponential backoff with a maximum delay of 60 seconds
                elif response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content'].strip()
                else:
                    print(f"Error: {response.status} {await response.text()}")
                    return "Error"
        except Exception as e:
            print(f"Exception: {e}")

    return "Error"  # Return error if all attempts fail

async def get_responses(prompts, model="gpt-4o", temperature=0.7, max_tokens=4096):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_response(session, prompt, model=model, temperature=temperature, max_tokens=max_tokens) for prompt in prompts]
        return await asyncio.gather(*tasks)

def get_batches(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

async def process_batch(batch, model, temperature, max_tokens, batch_delay):
    responses = await get_responses(batch, model, temperature, max_tokens)
    await asyncio.sleep(batch_delay)  # Add delay between batches to handle rate limits
    return responses

DOWNLOAD_DIR = '/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Dataset/Data'

def setup_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("API key for OpenAI is not set in the environment variables.")
    openai.api_key = api_key

def ensure_index(quotation_text,specimen_policy_text):
    documents = SimpleDirectoryReader(DOWNLOAD_DIR).load_data()
    embed_model = OpenAIEmbedding()
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    pipeline = IngestionPipeline(
        transformations=[
            splitter,
            embed_model,
        ]
    )
    
    nodes = pipeline.run(documents=documents)

    return VectorStoreIndex(nodes)

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
  
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

def rag_pipe(quotation_text, specimen_policy_text, user_extract_query,index):
    try:
        setup_openai_api_key()
        documents = SimpleDirectoryReader(DOWNLOAD_DIR+"/"+str(index)).load_data()
        embed_model = OpenAIEmbedding()
        Settings.embed_model = OpenAIEmbedding()

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)

        pipeline = IngestionPipeline(
            transformations=[
                splitter,
                embed_model,
            ]
        )
        
        nodes = pipeline.run(documents=documents)

        index= VectorStoreIndex(nodes)
        llm = OpenAiLlm(model="gpt-4o", max_tokens=4096)
        user_query = user_extract_query
        query_bundle = QueryBundle(user_query)

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=35,
        )
        
        retrieved_nodes = retriever.retrieve(query_bundle)
        reranker = LLMRerank(
            choice_batch_size=5,
            top_n=35,
        )
        
        
        response = reranker.postprocess_nodes(retrieved_nodes, query_bundle)
        dataset=[]
        for node_with_score in response:

            dataset.append(node_with_score.node.text)

        prompt = generate_prompt(user_extract_query, str(dataset))
        return(prompt,dataset)
    except Exception as e:
        print(e)

        return "There is no question return 'ERROR'",["ERROR"]

file_path='/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Dataset/dataset_coverage_fin_all.csv'
df = pd.read_csv(file_path)
yu=[]
source=[]
for index, row in df.iterrows():
    start_time = time.time()
    p1,s1=rag_pipe('','',row['Extracted User Query'],str(index))
    end_time = time.time()
    print(f"Time taken for row {index}: {end_time - start_time:.2f} seconds")
    yu.append(p1)
    source.append(s1)
responses = []
batches = list(get_batches(yu, 100))

async def process_all_batches():
    for i, batch in enumerate(batches):
        print(f"Processing batch {i + 1}/{len(batches)}...")
        batch_responses = await process_batch(batch, "gpt-4o", 0.7, 200, 0)
        responses.extend(batch_responses)
        print(f"Processed {len(responses)} entries out of {len(df)}")

asyncio.run(process_all_batches())

# Add the responses to the DataFrame
df['answer2'] = responses
df['sources']=source
df.to_csv('output_csv.csv', index=False)



