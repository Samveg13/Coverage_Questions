import os
import openai
import re
from ragatouille import RAGPretrainedModel
from openai import OpenAI
from prompts import SYSTEM_PROMPT, generate_prompt

def initialize_openai(api_key):
    openai.api_key = api_key

def extract_content(response_str):
    match = re.search(r"content='(.*?)', role=", response_str)
    if match:
        content = match.group(1)
        return content.replace("\\'", "'")
    return None

def get_chat_response(prompt, model="gpt-4o", temperature=0.7, max_tokens=1024):
    try:
        client = OpenAI()

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return extract_content(str(completion.choices[0].message))
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_rag_and_gpt(api_key, email):
    initialize_openai(api_key)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "/Users/samveg.shah/Desktop/Llama-index/.ragatouille/colbert/indexes/my_index")

    RAG = RAGPretrainedModel.from_index(index_path)
    results = RAG.search(query=email, k=10)
    
    context = "\n".join([x['content'] for x in results])
    

    #uncomment this to print the source passages

    # for x in results:
    #     print(x['content'])
    #     print("\n")
    #     print("===================")
    
    query = generate_prompt(email, context)
    
    client = OpenAI()
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        max_tokens=1024,
    )

    print(completion.choices[0].message)

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    email = """
    For our cyber insurance policy is it required we have a firewall? We had one installed by Teleco and are wanting to cancel it but unsure if this affects our cyber insurance? Can you please advise if and how this affects policy/coverage/premium.
    """
    test_rag_and_gpt(api_key, email)