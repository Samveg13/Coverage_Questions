import requests
import fitz  # PyMuPDF
import os
import json
import openai
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts_2 import SYSTEM_PROMPT, generate_prompt
from openai import OpenAI

from emails import data  # Import the sample_json_obj here


from aws_requests_auth.aws_auth import AWSRequestsAuth
from prompts import CHAT_TEXT_QA_PROMPT

DOWNLOAD_DIR = '/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Testing'



def setup_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("API key for OpenAI is not set in the environment variables.")
    openai.api_key = api_key

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
    response.raise_for_status()
    return response.json()



def get_policyholder_name(response):
    return response['extracted_data']['policyholder_name']


def make_authenticated_get_request(url, headers, params=None):
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
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


def download_pdf(url, filename, index):
    response = requests.get(url)
    file_path = os.path.join(DOWNLOAD_DIR + "/" + str(index) + '/', filename)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    return file_path


def via_cpn(cpn, index):
    try:
        api_url = 'https://admin-tools-gateway.coalitioninc.com/api/packages/internal'
        params = {'coalition_policy_number': cpn}
        headers = {
            'Authorization': 'Bearer '+ os.getenv("headers")
        }
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()
        response_json = response.json()

        doc_quotation = ""
        doc_specimen_policy = ""

        for document in response_json[0]['documents']:
            if document['label'] == 'Quotation':
                doc_quotation = document['link']
            elif document['label'] == 'Specimen Policy':
                doc_specimen_policy = document['link']

        if not doc_quotation or not doc_specimen_policy:
            return False, False

        quotation_path = download_pdf(doc_quotation, "quotation.pdf", index)
        specimen_policy_path = download_pdf(doc_specimen_policy, "specimen_policy.pdf", index)


        return quotation_path, specimen_policy_path
    except Exception as e:
        print(f"Failed due to {str(e)}")
        return False, False


def rag_pipe(quotation_text, specimen_policy_text, user_extract_query, index):
    try:
        return "response", "str(dataset)"
    except Exception as e:
        print(f"Failed due to {str(e)}")
        return False, False


def process_tickets(file_path):
    # Load the original dataframe
    df = pd.read_csv(file_path)[1:20]
    
    # Filter dataframe based on specified indices

    creds = load_credentials('/Users/samveg.shah/.aws/temp_creds.json')
    auth = create_aws_auth(creds)

    total_processed = 0
    success_count = 0
    aggregate_failures = {
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

    # Using ThreadPoolExecutor for concurrent execution
    with ThreadPoolExecutor(max_workers=100) as executor:
        future_to_row = {executor.submit(process_row, index, row, creds, auth): row for index, row in df.iterrows()}

        for future in as_completed(future_to_row):
            result_row, failures = future.result()
            responses.append(result_row)
            total_processed += 1
            success_count += result_row['response_lama'] != ''
            for key in aggregate_failures:
                aggregate_failures[key] += failures[key]
    
    # Save responses to file
    pd.DataFrame(responses).to_csv('/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Generated/output22.csv', index=False)

    print(f"\nTotal Processed Rows: {total_processed}")
    print(f"Total Successful Rows: {success_count}")
    for step, count in aggregate_failures.items():
        print(f"Failures at {step}: {count}")

def process_row(index, row, creds, auth):
    responses = []
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

    print(f"Processing index {index}")

    folder_path = os.path.join(DOWNLOAD_DIR, str(index))
    os.makedirs(folder_path, exist_ok=True)

    try:
        sample_json_obj = {
            "ticket_id": row['TICKET_ID'],
            "subject": row['SUBJECT'],
            "description": row['DESCRIPTION']
        }


        url = "https://mr5t4bsp2c.execute-api.us-west-2.amazonaws.com/dev/llm-relay-services/customer_success/ticket/classify_extract"

        try:
            
            response = make_authenticated_post_request(url, auth, sample_json_obj)

            policy_holder_name = get_policyholder_name(response)
        except Exception as e:
            if row['COALITION_POLICY_NUMBER'] is not None:

                quotation_text, specimen_policy_text = via_cpn(row['COALITION_POLICY_NUMBER'], str(index))
                if not quotation_text or not specimen_policy_text:
                    failures["aws_step"] += 1
                    row['response_lama'] = ''
                    row['source_lama'] = ''
                    return row, failures
                else:
                    response, source = rag_pipe(quotation_text, specimen_policy_text, row['Extracted User Query'], str(index))
                    if not response:
                        failures["Index Creation and Query Step"] += 1
                        row['response_lama'] = ''
                        row['source_lama'] = ''
                        return row, failures
                    else:
                        failures['Handles via cpn_aws_step'] += 1
                        row['response_lama'] = response
                        row['source_lama'] = source
                        return row, failures
            else:
                row['response_lama'] = ''
                row['source_lama'] = ''
                failures["aws_step"] += 1
                return row, failures

        headers = {
            'Authorization': 'Bearer '+ os.getenv("headers")
        }

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
        except Exception as e:
            if row['COALITION_POLICY_NUMBER'] is not None:

                quotation_text, specimen_policy_text = via_cpn(row['COALITION_POLICY_NUMBER'], str(index))
                if not quotation_text or not specimen_policy_text:
                    failures["step2_failure"] += 1
                    row['response_lama'] = ''
                    row['source_lama'] = ''
                    return row, failures
                else:
                    response, source = rag_pipe(quotation_text, specimen_policy_text, row['Extracted User Query'], str(index))
                    if not response:
                        failures["Index Creation and Query Step"] += 1
                        row['response_lama'] = ''
                        row['source_lama'] = ''
                        return row, failures
                    else:
                        failures['Handles via cpn_step2_failure'] += 1
                        row['response_lama'] = response
                        row['source_lama'] = source
                        return row, failures
            else:
                row['response_lama'] = ''
                row['source_lama'] = ''
                failures["step2_failure"] += 1
                return row, failures

        try:
            account_url = f'https://admin-tools-gateway.coalitioninc.com/account/{policy_id}'
            account_response = make_authenticated_get_request(account_url, headers)
            resp_cyber = account_response['cyber']
            uuid = get_policy_uuid(resp_cyber)
        except Exception as e:
            if row['COALITION_POLICY_NUMBER'] is not None:

                quotation_text, specimen_policy_text = via_cpn(row['COALITION_POLICY_NUMBER'], str(index))
                if not quotation_text or not specimen_policy_text:
                    failures["step3_failure"] += 1
                    row['response_lama'] = ''
                    row['source_lama'] = ''
                    return row, failures
                else:
                    response, source = rag_pipe(quotation_text, specimen_policy_text, row['Extracted User Query'], str(index))
                    if not response:
                        failures["Index Creation and Query Step"] += 1
                        row['response_lama'] = ''
                        row['source_lama'] = ''
                        return row, failures
                    else:
                        failures['Handles via cpn_step3_failure'] += 1
                        row['response_lama'] = response
                        row['source_lama'] = source
                        return row, failures
            else:
                row['response_lama'] = ''
                row['source_lama'] = ''
                failures["step3_failure"] += 1
                return row, failures

        try:
            documents_url = f'https://admin-tools-gateway.coalitioninc.com/api/packages/{uuid}/internal'
            response_pdf = make_authenticated_get_request(documents_url, headers)['documents']
            doc_quote, doc_specimen = extract_documents_data(response_pdf)
        except Exception as e:
            if row['COALITION_POLICY_NUMBER'] is not None:

                quotation_text, specimen_policy_text = via_cpn(row['COALITION_POLICY_NUMBER'], str(index))
                if not quotation_text or not specimen_policy_text:
                    failures["step4_failure"] += 1
                    row['response_lama'] = ''
                    row['source_lama'] = ''
                    return row, failures
                else:
                    response, source = rag_pipe(quotation_text, specimen_policy_text, row['Extracted User Query'], str(index))
                    if not response:
                        failures["Index Creation and Query Step"] += 1
                        row['response_lama'] = ''
                        row['source_lama'] = ''
                        return row, failures
                    else:
                        failures['Handles via cpn_step4_failure'] += 1
                        row['response_lama'] = response
                        row['source_lama'] = source
                        return row, failures
            else:
                row['response_lama'] = ''
                row['source_lama'] = ''
                failures["step4_failure"] += 1
                return row, failures

        try:
            specimen_policy_text = download_pdf(doc_quote, "quotation.pdf", str(index))
            quotation_text = download_pdf(doc_specimen, "specimen_policy.pdf", str(index))
        except Exception as e:
            row['response_lama'] = ''
            row['source_lama'] = ''
            failures["Text Extraction Step"] += 1
            return row, failures

        response, source = rag_pipe(quotation_text, specimen_policy_text, row['Extracted User Query'], str(index))
        if not response:
            failures["Index Creation and Query Step"] += 1
            row['response_lama'] = ''
            row['source_lama'] = ''
            return row, failures
        else:
            row['response_lama'] = response
            row['source_lama'] = source
            return row, failures

    except Exception as e:
        print(f"Failed processing ticket {row['TICKET_ID']} due to {str(e)}")
        return row, failures

if __name__ == "__main__":
    process_tickets(
        '/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Dataset/dataset_coverage_fin_all.csv'
    )




