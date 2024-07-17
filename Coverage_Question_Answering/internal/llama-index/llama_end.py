import requests
import fitz  # PyMuPDF
import os
import json
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

DOWNLOAD_DIR = '/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/internal/llama-index/documents'
def setup_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("API key for OpenAI is not set in the environment variables.")
    openai.api_key = api_key

def ensure_index(quotation_text,specimen_policy_text):
    documents = SimpleDirectoryReader("/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/internal/llama-index/documents").load_data()
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

def extract_pdf_text(url):
    pdf_response = requests.get(url)
    if pdf_response.status_code == 200:
        pdf_document = fitz.open(stream=pdf_response.content, filetype="pdf")
        return "".join([page.get_text() for page in pdf_document.pages()])
    return f"Failed to fetch PDF document. Status code: {pdf_response.status_code}"

def load_credentials(file_path):
    with open(file_path) as f:
        return json.load(f)

def create_aws_auth(creds):
    return AWSRequestsAuth(
    aws_access_key=os.getenv("aws_access_key"),
    aws_secret_access_key=os.getenv("aws_secret_access_key"),
    aws_token=os.getenv("aws_token"),
    aws_host=os.getenv("aws_host"),
    aws_region=os.getenv("aws_region"),
    aws_service=os.getenv("aws_service")
)

def make_authenticated_post_request(url, auth, data):
    response = requests.post(url, auth=auth, json=data)
    return response.json()

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
    
def get_policyholder_name(response):
    return response['extracted_data']['policyholder_name']

def make_authenticated_get_request(url, headers, params=None):
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def get_policy_uuid(cyber_data):
    for policy in cyber_data:
        if policy['lifecycle_state'] == "SIGNED_POLICY":
            return policy['uuid']
    raise ValueError("Error: no uuid found")

def extract_documents_data(response_pdf):
    doc_quote = next((doc['link'] for doc in response_pdf if doc['pdf_type'] == "QUOTATION"), "")
    doc_specimen = next((doc['link'] for doc in response_pdf if doc['pdf_type'] == "SPECIMEN_POLICY"), "")
    return doc_quote, doc_specimen

def download_pdf(url, filename):
    response = requests.get(url)
    file_path = os.path.join(DOWNLOAD_DIR, filename)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    return DOWNLOAD_DIR+"/"+str(filename)

def via_cpn(cpn):

    try:
        api_url = 'https://admin-tools-gateway.coalitioninc.com/api/packages/internal'
        params = {'coalition_policy_number': cpn}
        headers = {
            'Authorization': 'Bearer '+ os.getenv("headers")
        }
        response = requests.get(api_url, params=params, headers=headers)
        response_json = response.json()
        print(response_json)

        
        doc_quotation = ""
        doc_specimen_policy = ""

        for document in response_json[0]['documents']:
            if document['label'] == 'Quotation':
                doc_quotation = document['link']
            elif document['label'] == 'Specimen Policy':
                doc_specimen_policy = document['link']
        if not doc_quotation or not doc_specimen_policy:
            return False, False
        quotation_path = download_pdf(doc_quotation, "quotation.pdf")
        specimen_policy_path = download_pdf(doc_specimen_policy, "specimen_policy.pdf")
        
        return quotation_path, specimen_policy_path
    except Exception as e:
        print(f"Failed due to {str(e)}")
        return False, False

def rag_pipe(quotation_text, specimen_policy_text, user_extract_query):
    try:
        setup_openai_api_key()
        documents = SimpleDirectoryReader("/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/internal/llama-index/documents").load_data()
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
        dataset = [node_with_score.node.text for node_with_score in retrieved_nodes]
  
        prompt = generate_prompt(user_extract_query, str(response))
        response = get_chat_response(prompt)
        
        

        

        return response, str(dataset)
    except Exception as e:
        print(e)

        return False,False




def process_tickets(file_path):
    df = pd.read_csv(file_path)[1:2]

    creds = load_credentials('/Users/samveg.shah/.aws/temp_creds.json')
    auth = create_aws_auth(creds)

    success_count = 0
    total_processed = 0
    failures = {
        "aws_step": 0,
        "step2_failure": 0,
        "step3_failure": 0,
        "step4_failure": 0,
        "Text Extraction Step": 0,
        "Index Creation and Query Step": 0,
        "Handles via cpn_aws_step": 0,
        "Handles via cpn_step2_failure": 0,
        "Handles via cpn_step3_failure": 0,
        "Handles via cpn_step4_failure": 0,
    }

    responses = []

    for index, row in df.iterrows():
        print(row['COALITION_POLICY_NUMBER'])

        print(f"Processing index {index}")
        total_processed += 1
        try:
            # Prepare sample_json_obj with values from Excel
            sample_json_obj = {
                "ticket_id": row['TICKET_ID'],
                "subject": row['SUBJECT'],
                "description": row['DESCRIPTION']
            }

            url = "https://mr5t4bsp2c.execute-api.us-west-2.amazonaws.com/dev/llm-relay-services/customer_success/ticket/classify_extract"

            # Step 1: Make authenticated post request
            try:
                response = make_authenticated_post_request(url, auth, sample_json_obj)
                policy_holder_name = get_policyholder_name(response)
                print("AWS step success")
            except Exception as e:
                print("AWS step failure")
                if str(row['COALITION_POLICY_NUMBER']) !='nan':
                    quotation_text, specimen_policy_text = via_cpn(row['COALITION_POLICY_NUMBER'])
                    if quotation_text == False or specimen_policy_text == False:
                        failures["aws_step"] += 1
                        row['response_lama'] = ''
                        row['source_lama'] = ''
                        responses.append(row)
                        
                        continue
                    else:
                        response,source = rag_pipe(quotation_text, specimen_policy_text, row['Extracted User Query'])
                        if response == False:
                            failures["Index Creation and Query Step"] += 1
                            row['response_lama'] = 'None'
                            row['source_lama'] = 'None'
                            continue
                        else:
                            failures['Handles via cpn_aws_step'] += 1
                            success_count += 1
                            row['response_lama'] = response
                            row['source_lama'] = response
                            responses.append(row)
                            continue
                else:
                    row['response_lama'] = ''
                    row['source_lama'] = ''
                    responses.append(row)
                    failures["aws_step"] += 1
                    continue

            headers = {
            'Authorization': 'Bearer '+ os.getenv("headers")
        }

            # Step 2: Search for the policy by name
            try:
                search_url = 'https://platform-search.coalitioninc.com/search/internal'
                params = {
                    'q': policy_holder_name,
                    'facet': 'account',
                    'offset': 0,
                    'limit': 1
                }
                response_json_step2 = make_authenticated_get_request(search_url, headers, params)
                policy_id = response_json_step2['account']['data'][0]['_source']['id']
                print("Step 2 success")
            except Exception as e:
                print("Step 2 failure")
                if str(row['COALITION_POLICY_NUMBER']) !='nan':
                    quotation_text, specimen_policy_text = via_cpn(row['COALITION_POLICY_NUMBER'])
                    if quotation_text == False or specimen_policy_text == False:
                        failures["step2_failure"] += 1
                        row['response_lama'] = ''
                        row['source_lama'] = ''
                        responses.append(row)
                        continue
                    else:
                        response,source = rag_pipe(quotation_text, specimen_policy_text, row['Extracted User Query'])
                        if response == False:
                            failures["Index Creation and Query Step"] += 1
                            row['response_lama'] = ''
                            row['source_lama'] = ''
                            responses.append(row)
                            continue
                        else:
                            failures['Handles via cpn_step2_failure'] += 1
                            success_count += 1
                            row['response_lama'] = response
                            row['source_lama'] = source
                            
                            responses.append(row)
                            continue
                else:
                    row['response_lama'] = ''
                    row['source_lama'] = ''
                    responses.append(row)
                    failures["step2_failure"] += 1
                    continue

            # Step 3: Retrieve account details
            try:
                account_url = f'https://admin-tools-gateway.coalitioninc.com/account/{policy_id}'
                account_response = make_authenticated_get_request(account_url, headers)
                resp_cyber = account_response['cyber']
                uuid = get_policy_uuid(resp_cyber)
                print("Step 3 success")
            except Exception as e:
                print("Step 3 failure")
                if str(row['COALITION_POLICY_NUMBER']) !='nan':
                    quotation_text, specimen_policy_text = via_cpn(row['COALITION_POLICY_NUMBER'])
                    if quotation_text == False or specimen_policy_text == False:
                        row['response_lama'] = ''
                        row['source_lama'] = ''
                        responses.append(row)
                        failures["step3_failure"] += 1
                        continue
                    else:
                        response,source = rag_pipe(quotation_text, specimen_policy_text, row['Extracted User Query'])
                        if response == False:
                            failures["Index Creation and Query Step"] += 1
                            row['response_lama'] = ''
                            row['source_lama'] = ''
                            responses.append(row)
                            continue
                        else:
                            failures['Handles via cpn_step3_failure'] += 1
                            success_count += 1
                            row['response_lama'] = response
                            row['source_lama'] = source
                            responses.append(row)
                            continue
                else:
                    row['response_lama'] = ''
                    row['source_lama'] = ''
                    responses.append(row)
                    failures["step3_failure"] += 1
                    continue

            # Step 4: Retrieve policy documents data
            try:
                documents_url = f'https://admin-tools-gateway.coalitioninc.com/api/packages/{uuid}/internal'
                response_pdf = make_authenticated_get_request(documents_url, headers)['documents']
                doc_quote, doc_specimen = extract_documents_data(response_pdf)
                print("Step 4 success")
            except Exception as e:
                print("Step 4 failure")
                if str(row['COALITION_POLICY_NUMBER']) !='nan':
                    quotation_text, specimen_policy_text = via_cpn(row['COALITION_POLICY_NUMBER'])
                    if quotation_text == False or specimen_policy_text == False:
                        failures["step4_failure"] += 1
                        row['response_lama'] = ''
                        row['source_lama'] = ''
                        responses.append(row)
                        continue
                    else:
                        response,source = rag_pipe(quotation_text, specimen_policy_text, row['Extracted User Query'])
                        if response == False:
                            failures["Index Creation and Query Step"] += 1
                            row['response_lama'] = ''
                            row['source_lama'] = ''
                            responses.append(row)
                            continue
                        else:
                            failures['Handles via cpn_step4_failure'] += 1
                            success_count += 1
                            row['response_lama'] = response
                            row['source_lama'] = source
                            responses.append(row)
                            continue
                else:
                    row['response_lama'] = ''
                    row['source_lama'] = ''
                    responses.append(row)
                    failures["step4_failure"] += 1
                    continue

            # Text Extraction
            try:
                specimen_policy_text = download_pdf(doc_quote, "quotation.pdf")
                quotation_text = download_pdf(doc_specimen, "specimen_policy.pdf")
                print("Text extracted")
            except Exception as e:
                row['response_lama'] = ''
                row['source_lama'] = ''
                responses.append(row)
                failures["Text Extraction Step"] += 1
                continue

            # Ensure index is created with extracted text
            response,source = rag_pipe(quotation_text, specimen_policy_text, row['Extracted User Query'])
            if response == False:
                failures["Index Creation and Query Step"] += 1
                row['response_lama'] = ''
                row['source_lama'] = ''
                responses.append(row)
                continue
            else:
                success_count += 1
                row['response_lama'] = response
                row['source_lama'] = source
                responses.append(row)
                print("Query Response:", response)
        except Exception as e:
            print(f"Failed processing ticket {row['ticket_id']} due to {str(e)}")

    pd.DataFrame(responses).to_csv('/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Generated/output22.csv', index=False)

    print(f"\nTotal Processed Rows: {total_processed}")
    print(f"Total Successful Rows: {success_count}")
    for step, count in failures.items():
        print(f"Failures at {step}: {count}")

if __name__ == "__main__":
    process_tickets('/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Dataset/dataset_coverage_fin_all.csv')





