import openai

def initialize_openai(api_key):
    openai.api_key = api_key

def get_chat_response(prompt, model="gpt-4", temperature=0.7, max_tokens=1024):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example Usage
if __name__ == "__main__":
    api_key = "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"
    initialize_openai(api_key)
    
    prompt = """ This is an email or thread of email between the cyber insruance provider named Coalition and the client, in the following email the client has asked a query, question or seeks clariofication on something. Can you extract the part where that is. Make sure to extract the entire part and not just some lines of the query, the query might be spaced out on a number of lines The email is: "Hi Katie or Brian,

 

    "Thank you

 

Is there any chance of getting a $10,000 deductible on the $1MM options

 

 

 

 

I am away on holidays August 2nd, 2024 to August 26th, 2024

 

 

HUB International Insurance Brokers

Risk & Insurance | Employee Benefits | Retirement & Private Wealth

Ready for tomorrow.

 

Otilja Majewski

Sr. Commercial Marketing Manager

HUB International Limited

Suite 101, 9906-106 Street

Grande Prairie, ​AB ​T8V 6L6

Direct : (780)833-9990

Cell : (780)402-9396

Otilja.majewski@hubinternational.com

hubinternational.com (http://www.hubinternational.com/)

Privacy Notice: To view our updated Privacy Policy visit https://www.hubinternational.com/en-CA/ and click on the word PRIVACY at the bottom of the webpage.

 

This communication (and any information or material transmitted with this communication) is confidential, may be privileged and is intended only for the use of the intended recipient. If you are not the intended recipient, any review, retransmission, conversion to hard copy, copying, circulation, publication, dissemination, distribution, reproduction or other use of this communication, information or material is strictly prohibited and may be illegal. If you received this communication in error, please notify us immediately by telephone or by return email, and delete this communication, information and material from any computer, disk drive, diskette or other storage device or media. Thank you.

 

From: Coalition <join@coalitioninc.com>
Sent: Thursday, June 13, 2024 10:32 AM
To: Majewski, Otilja <otilja.majewski@hubinternational.com>
Cc: Joel Lauzon <joel@coalitioninc.com>
Subject: [EXTERNAL] Quotes for Oculus Transport Ltd. a/o Ric...

 

 

Dear Otilja Majewski,

Thank you for your request! We've created 3 quote(s) for Oculus Transport Ltd. a/o Ric Peterson Developments Inc. outlined below. Please find attached the multiple quote comparison sheet, quote document(s), Coalition Risk Assessment, specimen policy, and signature bundle(s). You may view the quote details and access these documents at any point from your Coalition Broker Dashboard. Please see below coverage for binding instructions with effective date: June 14, 2024.

Limit

Retention

Total
Price

Status

$1M

$25K

$9,365

Active

$2M

$30K

$13,570

Active

$3M

$35K

$17,388

Active

Bind Now  (https://mandrillapp.com/track/click/31017443/platform.coalitioninc.com?p=eyJzIjoib3RxY0o4SkM3aElUYWlOWWVxeWFRV2FzRXFFIiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL3BsYXRmb3JtLmNvYWxpdGlvbmluYy5jb21cXFwvYXBwXFxcL2FjY291bnRzXFxcLzhiM2I5YTI5LWFkNGEtNGQ4YS04YmFiLTYyNmMwNzAyZTcwMj90eXBlPWN5YmVyJnNob3c9YjZiMTFjZTEtOTdjMi00OWIyLTkwOGQtNmRlNWIyNmU5YWY5XCIsXCJpZFwiOlwiZDZhZTU0YTQ4Njc0NGZjNThiZGJlMzc3YmNkYTA4NDBcIixcInVybF9pZHNcIjpbXCI3NTUyZGM1YzRkZjI4OGYxY2Q0OTM3YTljMjI4YWEzYjk0Y2NjMWNkXCJdfSJ9)

View Quotes  (https://mandrillapp.com/track/click/31017443/platform.coalitioninc.com?p=eyJzIjoib3RxY0o4SkM3aElUYWlOWWVxeWFRV2FzRXFFIiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL3BsYXRmb3JtLmNvYWxpdGlvbmluYy5jb21cXFwvYXBwXFxcL2FjY291bnRzXFxcLzhiM2I5YTI5LWFkNGEtNGQ4YS04YmFiLTYyNmMwNzAyZTcwMj90eXBlPWN5YmVyJnNob3c9YjZiMTFjZTEtOTdjMi00OWIyLTkwOGQtNmRlNWIyNmU5YWY5XCIsXCJpZFwiOlwiZDZhZTU0YTQ4Njc0NGZjNThiZGJlMzc3YmNkYTA4NDBcIixcInVybF9pZHNcIjpbXCI3NTUyZGM1YzRkZjI4OGYxY2Q0OTM3YTljMjI4YWEzYjk0Y2NjMWNkXCJdfSJ9)



How to bind this account
There are two ways to bind one of these quotes.

1. You may bind coverage from your Coalition Broker Dashboard (https://mandrillapp.com/track/click/31017443/platform.coalitioninc.com?p=eyJzIjoiSk1sQ1h6NHZHdHcwMkNJdk5xLXZyN2V5bUIwIiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwOlxcXC9cXFwvcGxhdGZvcm0uY29hbGl0aW9uaW5jLmNvbVwiLFwiaWRcIjpcImQ2YWU1NGE0ODY3NDRmYzU4YmRiZTM3N2JjZGEwODQwXCIsXCJ1cmxfaWRzXCI6W1wiNTg0NDZjNDFmZTUwOGQ0MGYyZjJkNjM1MTIwMjAyYjMwMjkzMjViZVwiXX0ifQ) , which triggers a digital signature from your client (you’ll receive a copy, too).
Or,
2. You may return the signature bundle attached here. All documents must be signed to issue the policy.


All Coalition policies come with access to technology that brokers and policyholders can use to reduce digital risks before it strikes. We refer to this as Active Insurance from Coalition which includes coverage designed to prevent digital risk before it strikes on top of the traditional aspects of coverage you are already getting from a top tier insurer.

Thank you again!
We appreciate the opportunity to work with you as a risk management partner to Oculus Transport Ltd. a/o Ric Peterson Developments Inc.. If you have any questions, either regarding this quote - or to learn about how Active Insurance from Coalition provides continuous protection from digital risks - please contact us at help@coalitioninc.com (https://mandrillapp.com/track/click/31017443/www.coalitioninc.com?p=eyJzIjoiLXhHbmlNNlphbW5RaXBuWk1iYzhmU2RFM0pZIiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL3d3dy5jb2FsaXRpb25pbmMuY29tXFxcL2VuLWNhXFxcL2NvbnRhY3RcIixcImlkXCI6XCJkNmFlNTRhNDg2NzQ0ZmM1OGJkYmUzNzdiY2RhMDg0MFwiLFwidXJsX2lkc1wiOltcImNhNmFjOTE4YTE1MjEwODYyNWE3YzlkYmNmNzdiMzhhZTMxN2IzZDNcIl19In0) .

All the best,
Coalition Team

 

Sent by Coalition Inc. (https://mandrillapp.com/track/click/31017443/www.coalitioninc.com?p=eyJzIjoiUGM4SmZ1VEVaazJRd3dvbTRPUmpiaXNrT3FvIiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL3d3dy5jb2FsaXRpb25pbmMuY29tXFxcL1wiLFwiaWRcIjpcImQ2YWU1NGE0ODY3NDRmYzU4YmRiZTM3N2JjZGEwODQwXCIsXCJ1cmxfaWRzXCI6W1wiMjgzMWYwY2QyODc4OGMzYTAzNDE3OTFkZGI1MzVlYTcxOTk0ZmYyNFwiXX0ifQ) 55 2nd Street, Suite 2500, San Francisco CA 94105

Help Center (https://mandrillapp.com/track/click/31017443/www.coalitioninc.com?p=eyJzIjoiTHBqM1A5dlpJdnVqTFFUTWR6MEc3b3UybEVBIiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL3d3dy5jb2FsaXRpb25pbmMuY29tXFxcL2xlYXJuaW5nXCIsXCJpZFwiOlwiZDZhZTU0YTQ4Njc0NGZjNThiZGJlMzc3YmNkYTA4NDBcIixcInVybF9pZHNcIjpbXCI4Y2JiMDlmZDMyZmVhNmY2N2UxYzU3Y2FlNWYyYTA4MGE2MWMxNjQ0XCJdfSJ9) • Privacy Policy (https://mandrillapp.com/track/click/31017443/www.coalitioninc.com?p=eyJzIjoiaVlpQWNKMWtldTRpNzh0N21xOUZFNDdzUFRjIiwidiI6MSwicCI6IntcInVcIjozMTAxNzQ0MyxcInZcIjoxLFwidXJsXCI6XCJodHRwczpcXFwvXFxcL3d3dy5jb2FsaXRpb25pbmMuY29tXFxcL2xlZ2FsXFxcL3ByaXZhY3lcIixcImlkXCI6XCJkNmFlNTRhNDg2NzQ0ZmM1OGJkYmUzNzdiY2RhMDg0MFwiLFwidXJsX2lkc1wiOltcIjVkODIzNjc5MTI3NTcyYTFiYjFlMjhlYjc4NGZjZDUzYjM3MmI3YjRcIl19In0)"""
    response = get_chat_response(prompt)
    print("Response:", response)


#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