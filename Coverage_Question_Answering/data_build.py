
import openai
import pandas as pd
import aiohttp
import asyncio
import time
import os
def initialize_openai(api_key):
    openai.api_key = 

async def fetch_response(session, prompt, retry_attempts=5, model="gpt-4o", temperature=0.7, max_tokens=4096):
    url = ""
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
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

def process_emails(input_csv, output_csv, api_key, batch_size=500, batch_delay=0):
    # Initialize OpenAI API Key
    initialize_openai(api_key)

    # Read the CSV file with enhanced error handling
    try:
        df = pd.read_csv(input_csv, delimiter=',', quotechar='"')
    except pd.errors.ParserError as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Check if 'DESCRIPTION' column exists
    if 'DESCRIPTION' not in df.columns:
        print("The column 'DESCRIPTION' does not exist in the CSV file.")
        return

    # Create prompts for each email
    prompts = []
    for _, row in df.iterrows():
        prompt = f"""
        You have been given an email or a thread of emails containing user queries. Some of these emails can be answered using user specific policy documents or general company documents, while others are logistics-based emails that cannot be answered with the information present in the policy documents of that user.

        Style Criteria:
        1. Does X policy/quote/document include Y coverage(s)?
        2. Does Y coverage cover Z use case?
        3. Define Y coverage.
        4. Scenario based question where if Y happens will it be covered under any of the coverages
        5. Does X coverage have Y premium
        6. Can you tell me about X endorsement
        9. What is the limit of XYZ coverage?
        10. Will X fall under the policy/quote/document
        11. Does the policy/quote allow X?
        13. Any questions which inquires about coverage if some event happens
        14. Will X affect the coverage?



        X,Y,Z,XYZ are placeholders and can be any coverage,endorsement,limit etc. Any email which inquires 'whether a particular endorsement or coverage can be included in the policy or if they can get XYZ limit' does not fit into the above criteria.

     
        Classification Criteria:
        - If **at least one** question in the user query meets the above style criteria, classify the user query as "can be answered."
        - If none of the questions in the user query meet the above style criteria, classify the user query as "cannot be answered."


        Your task is to classify each email into one of two categories: 'Can be answered with documents' or 'Cannot be answered with documents.'
        For each email, provide the following:
        Category: Can be answered with documents / Cannot be answered with documents
        Justification:
        User Query under consideration:
        Here is the email: {row['DESCRIPTION']}





        """
        prompts.append(prompt)

    responses = []
    batches = list(get_batches(prompts, batch_size))

    async def process_all_batches():
        for i, batch in enumerate(batches):
            print(f"Processing batch {i + 1}/{len(batches)}...")
            batch_responses = await process_batch(batch, "gpt-4o", 0.7, 200, batch_delay)
            responses.extend(batch_responses)
            print(f"Processed {len(responses)} entries out of {len(df)}")

    asyncio.run(process_all_batches())

    # Add the responses to the DataFrame
    df['answer2'] = responses

    # Write the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Processing completed. {len(df)} entries saved to {output_csv}.")

if __name__ == "__main__":
    # Set your OpenAI API key here
    api_key =   # Replace with your actual API key
    print("hi")


    # Define input and output file paths
    input_csv = "/Users/samveg.shah/Downloads/taxonomy_workstype.csv"  # Ensure this is a valid CSV file path
    output_csv = "/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Generated/combined_output_full_segregated.csv"  # Change to your desired output file path

    # Process emails
    process_emails(input_csv, output_csv, api_key, batch_size=1500, batch_delay=0)
