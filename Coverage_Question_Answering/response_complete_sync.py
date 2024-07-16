import requests
import fitz  # PyMuPDF
import os
import json
import time
import openai
import pandas as pd
from llama_index.core import QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from prompts import SYSTEM_PROMPT, generate_prompt
from openai import OpenAI
from ragatouille.data import CorpusProcessor

from ragatouille import RAGPretrainedModel
from llama_index.core.schema import Document
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.llms.openai import OpenAI as OpenAiLlm
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor import LLMRerank

from aws_requests_auth.aws_auth import AWSRequestsAuth
import openai
import pandas as pd
import aiohttp
import asyncio
import time
import os
def initialize_openai(api_key):
    openai.api_key = api_key


def extract_pdf_text(file_path):
    try:
        pdf_document = fitz.open(file_path)
        text = "".join([page.get_text() for page in pdf_document])
        return text
    except Exception as e:
        return f"Failed to extract text from PDF document. Error: {e}"

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
    print(user_extract_query)
    try:
        # print(user_extract_query)
        # print(quotation_text)
        # print(specimen_policy_text)
        setup_openai_api_key()
        corpus_processor = CorpusProcessor()

        # RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        print()
        documents = [extract_pdf_text('/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Dataset/Data/'+index+'/quotation.pdf'),extract_pdf_text('/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Dataset/Data/'+index+'/specimen_policy.pdf')]
        print(documents)
        # documents = corpus_processor.process_corpus(documents, chunk_size=256)
        
        
        # index=RAG.encode([x['content'] for x in documents], document_metadatas=[{"about": "ghibli"} for _ in range(len(documents))])
        # docs=RAG.search_encoded_docs(query = user_extract_query, k=10)
        # context = "\n".join([x['content'] for x in docs])
        
        prompt = generate_prompt(user_extract_query, str(documents))
        print(prompt)
        return(prompt,'')
    except Exception as e:
        print(e)

        return "There is no question return 'ERROR'",["ERROR"]

file_path='/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Dataset/dataset_coverage_fin_all.csv'
df = pd.read_csv(file_path)
yu=[]
source=[]
for index, row in df.iterrows():
    print(index)
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
df.to_csv('output_csv2.csv', index=False)
