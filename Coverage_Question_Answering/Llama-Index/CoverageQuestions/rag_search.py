from ragatouille import RAGPretrainedModel
from openai import OpenAI
client = OpenAI()

main_query="""
        The email provided is : {email}
        The context is: {context}
        """

email = """Hello, The insured is wanting to know if the cyber policy covers the team anywhere? Even if they are in different offices, cities, homes, etc.
  


"""
RAG = RAGPretrainedModel.from_index("/Users/samveg.shah/Desktop/Llama-index/.ragatouille/colbert/indexes/my_index")
results = RAG.search(query=email,k=10)
context=[]
for x in results:
    context.append(x['content'])
    print(x['content'])
    print("\n")
    print("===================")
print(main_query.format(email=email,context=context))
completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": """You are an expert Q&A system that is trusted around the world.\n
        Always answer the query using the provided context information which are from policy documents, 
        and not prior knowledge.\n
        Some rules to follow:\n
        1. Directly reference the given context in your answer wherever necessary.\n
        2. Avoid statements like 'Based on the context, ...' or 
        'The context information ...' or anything along rather use terms like 'based on the policy documents'  \n
        3. Use phrases like “this may be the section that references your question” instead of “this will be covered\n
        4. Use words like, may and perhaps if needed where information is not clear \n
        5. Don’t use words that indicate a definitive coverage to avoid liability

        Please note: Some queries may be in form of a situation, use the context provided to answer the questions which follows the sitation and justify  """},
    {"role": "user", "content": main_query.format(email=email,context=context)}
  ],
  max_tokens=1024,

)

