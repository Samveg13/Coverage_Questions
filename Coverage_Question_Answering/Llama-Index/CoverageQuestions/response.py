import os
import openai
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
import sys
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.llms.openai import OpenAI as OpenAiLlm
import logging
from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate
# from llama_index.response.pprint_utils import pprint_response


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

# Either way we can now query the index
# shakespeare!

TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information which are from policy documents, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Directly reference the given context in your answer wherver necessary.\n"
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


llm = OpenAiLlm(model="gpt-4", max_tokens=1024)  # Ensure the correct model name
messages = [
    ChatMessage(
        role="system", content="You will be provided with an email/thread od emails and you responsibility is to extract the portion which has the user questions. The format should be assistant:....."
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

    Contingencies are often related to security issues but can also be related to custom endorsements or other concerns. Please note the terms listed below are non-binding and responses to the contingencies may change our terms. You’ll find the quote of insurance, signature bundle, specimen policy, and Coalition Risk Assessment attached to this email.

    View Quote  (https://mandrillapp.com/track/click/31017443/platform.coalitioninc.com?p=eyJzIjoiNnd5dnZfRVBvOVl4bDlvWjNac0RueW5SVTRRIiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL3BsYXRmb3JtLmNvYWxpdGlvbmluYy5jb21cXFwvYXBwXFxcL2FjY291bnRzXFxcLzFhYmJjZGZlLWU1YWItNDliMC1iNTg0LThlZjM3NTUwNGMwNz90eXBlPWN5YmVyJnNob3c9ZmY4NGM4ZWUtYTgxMS00OTUxLTgwYTAtYjYwZGFjNzIwZDNmXCIsXCJpZFwiOlwiYjgxNGM3ZTVlYzQzNDhmNDk5OGViZDM5ZTE0NDNkMjNcIixcInVybF9pZHNcIjpbXCI3NTUyZGM1YzRkZjI4OGYxY2Q0OTM3YTljMjI4YWEzYjk0Y2NjMWNkXCJdfSJ9)


    In order to bind this account, these contingencies must be completed prior to binding:

    * Confirm the volume of records the insured collects, processes, transmits, or has access to. This includes PII as defined by state and federal laws.


    Quote overview:

    * Limit: $5,000,000
    * Retention: $25,000
    * Premium: $24,987
    * Total Price: $24,987
    * Insurance Market: Surplus
    * Coverage Period: February 03, 2024 - February 03, 2025



    How to resolve the contingency and bind this quote
    We are ready to bind for you (or help you bind from your Coalition Broker Dashboard (https://mandrillapp.com/track/click/31017443/platform.coalitioninc.com?p=eyJzIjoiR2JQOV9xelBLaTB4TS05V3M4RFJZdlY0djI4IiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwOlxcXC9cXFwvcGxhdGZvcm0uY29hbGl0aW9uaW5jLmNvbVwiLFwiaWRcIjpcImI4MTRjN2U1ZWM0MzQ4ZjQ5OThlYmQzOWUxNDQzZDIzXCIsXCJ1cmxfaWRzXCI6W1wiNTg0NDZjNDFmZTUwOGQ0MGYyZjJkNjM1MTIwMjAyYjMwMjkzMjViZVwiXX0ifQ) ) as soon as the contingency is resolved. If you have any questions on the contingency, please schedule a call with our Security Experts:

    1. Go to this page https://info.coalitioninc.com/schedule-security-call.html (https://mandrillapp.com/track/click/31017443/info.coalitioninc.com?p=eyJzIjoiLXhyTFM5Vm9qbWRhc2NFVTFpVG1sSld2bDg0IiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL2luZm8uY29hbGl0aW9uaW5jLmNvbVxcXC9zY2hlZHVsZS1zZWN1cml0eS1jYWxsLmh0bWxcIixcImlkXCI6XCJiODE0YzdlNWVjNDM0OGY0OTk4ZWJkMzllMTQ0M2QyM1wiLFwidXJsX2lkc1wiOltcImE0YTIxNGE4NWE1MWNlNzdkODNjZjEyNWJkZGEyNTgxMjNkM2I1ZDNcIl19In0)
    2. Select your role
    3. Make sure to select the right reason related to 'contingency' in order to get routed to the right team


    Need help addressing a competitive quote?
    Our underwriting team is here to help. Please ""reply all"" to this email and add the competitive quote documentation as an attachment or provide premium, limit, and retention details. Coalition's underwriting team will review and get back to you shortly.

    What is Active Insurance?
    Active Insurance is coverage designed to prevent digital risk before it strikes. Unlike traditional insurance, designed only to cover and transfer risk when the worst happens, Active Insurance provides continual risk assessment, monitoring and response to address risks that move at digital speed.

    Take advantage of Active Insurance for D&O and EPL, too
    Coalition’s Active Executive Risks coverages can help protect organizations and their executives from evolving digital risks. Our approach combines the power of our active monitoring technology, real-time data analysis, and comprehensive insurance coverage to accelerate the quoting process and give your clients best-in-class protection. Check out our appetite guide (https://mandrillapp.com/track/click/31017443/cdn.intelligencebank.com?p=eyJzIjoiQVhwd1d4V0p4b3dZdFl1MFlRMnRicnRicE9BIiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL2Nkbi5pbnRlbGxpZ2VuY2ViYW5rLmNvbVxcXC91c1xcXC9zaGFyZVxcXC9OTVhEXFxcLzd6UjBcXFwvekcwZVxcXC9vcmlnaW5hbFxcXC9FeGVjdXRpdmUrUmlza3MrQXBwZXRpdGVcIixcImlkXCI6XCJiODE0YzdlNWVjNDM0OGY0OTk4ZWJkMzllMTQ0M2QyM1wiLFwidXJsX2lkc1wiOltcIjQyNWE1NWI3ODNmNjc3OGZiNzlhNzhmYTk1MzEyZWZmOThjYjA5ZTdcIl19In0) and start quoting (https://mandrillapp.com/track/click/31017443/platform.coalitioninc.com?p=eyJzIjoieFl0dFRMaDFIY2trY1J2UHRsc1hnd3hoQlhRIiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL3BsYXRmb3JtLmNvYWxpdGlvbmluYy5jb21cXFwvYXBwXFxcL3F1b3RlLWZsb3dcXFwvYXBwbGljYXRpb25cIixcImlkXCI6XCJiODE0YzdlNWVjNDM0OGY0OTk4ZWJkMzllMTQ0M2QyM1wiLFwidXJsX2lkc1wiOltcImFiNDU5Yjc2MDU1OGUzMDk1NDg4NDY2MWE3MDBjNWE2MDhmNzI4ODdcIl19In0) today.

    Thank you again!
    We appreciate the opportunity to work with you as a risk management partner to your clients. If you have any questions, either regarding this quote - or to learn about how Active Insurance from Coalition provides continuous protection from digital risks - please contact us at help@coalitioninc.com (https://mandrillapp.com/track/click/31017443/www.coalitioninc.com?p=eyJzIjoiU2gxZVl4aG12dWpaTVF2VW5nQjh4alRvSU00IiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL3d3dy5jb2FsaXRpb25pbmMuY29tXFxcL2NvbnRhY3RcIixcImlkXCI6XCJiODE0YzdlNWVjNDM0OGY0OTk4ZWJkMzllMTQ0M2QyM1wiLFwidXJsX2lkc1wiOltcImU2ZWMwM2JiYTk2YzY1ZDEzODcxODU0YTdhNGE3MGRmYTkyMzEyZmFcIl19In0) .

    All the best,
    Coalition Team

    

    Sent by Coalition Inc. (https://mandrillapp.com/track/click/31017443/www.coalitioninc.com?p=eyJzIjoiM2c0TEJfOU1yd0JCRVloQThyWWNQVDBwNG1ZIiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL3d3dy5jb2FsaXRpb25pbmMuY29tXFxcL1wiLFwiaWRcIjpcImI4MTRjN2U1ZWM0MzQ4ZjQ5OThlYmQzOWUxNDQzZDIzXCIsXCJ1cmxfaWRzXCI6W1wiMjgzMWYwY2QyODc4OGMzYTAzNDE3OTFkZGI1MzVlYTcxOTk0ZmYyNFwiXX0ifQ) 55 2nd Street, Suite 2500, San Francisco CA 94105

    Help Center (https://mandrillapp.com/track/click/31017443/www.coalitioninc.com?p=eyJzIjoiRlEwRTBLNEJlYml3bklBcTluSWU1TnF6bS1zIiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL3d3dy5jb2FsaXRpb25pbmMuY29tXFxcL2xlYXJuaW5nXCIsXCJpZFwiOlwiYjgxNGM3ZTVlYzQzNDhmNDk5OGViZDM5ZTE0NDNkMjNcIixcInVybF9pZHNcIjpbXCI4Y2JiMDlmZDMyZmVhNmY2N2UxYzU3Y2FlNWYyYTA4MGE2MWMxNjQ0XCJdfSJ9) • Privacy Policy (https://mandrillapp.com/track/click/31017443/www.coalitioninc.com?p=eyJzIjoiM0V0czJxZDBLR25ORWdKMW00NG56NXMtTmNNIiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL3d3dy5jb2FsaXRpb25pbmMuY29tXFxcL2xlZ2FsXFxcL3ByaXZhY3lcIixcImlkXCI6XCJiODE0YzdlNWVjNDM0OGY0OTk4ZWJkMzllMTQ0M2QyM1wiLFwidXJsX2lkc1wiOltcIjVkODIzNjc5MTI3NTcyYTFiYjFlMjhlYjc4NGZjZDUzYjM3MmI3YjRcIl19In0)

    This e-mail and any attachments may contain information that is privileged or confidential and is meant solely for the use of person(s) to whom it was intended to be addressed. If you have received this e-mail by mistake, or you are not the intended recipient, you are not authorized to read, print, keep, copy or distribute this message, attachments, or any part of the same. If you have received this email in error, please immediately inform the author and permanently delete the original, all copies and any attachments of this email from your computer. Thank you amwins.com.""")
    ]
resp = str(llm.chat(messages)).split(":")[1]
print("//////????")
# print("The llm is ")
print(resp)
# print("UOOOO")
resp=""" 
 Hello, do you have any cryptojacking included in your policy? 
"""
response=index.as_query_engine(
        text_qa_template=CHAT_TEXT_QA_PROMPT,
        llm=llm,
        similarity_top_k=5
    ).query(resp)

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

# Path to the new Python file where we'll write the source passages
# Placeholder for additional operations like saving the text to a file can be added here