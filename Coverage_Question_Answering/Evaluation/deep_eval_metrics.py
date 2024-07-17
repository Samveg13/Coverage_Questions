from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric,ContextualRecallMetric,AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
import pandas as pd
import ast


def custom_str_to_list(s):
    if isinstance(s, str) and s.startswith('[') and s.endswith(']'):
        items = s[1:-1].split(',')
        items = [item.strip().strip('"').strip("'") for item in items]
        return items
    return s
metrics=[]

file_path = '/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Generated/final_corrected.csv'  # Update this with the path to your CSV file
df = pd.read_csv(file_path)

for j,i in df.iterrows():

    if(i['lama correctness']=="Not present" or i['Gold Answer Present']=="No"):
        continue

    # Replace this with the actual output from your LLM application
    actual_output = i['AI_Response']

    # Replace this with the actual retrieved context from your RAG pipeline
    retrieval_context = [i['Source_Passages_AI']]
    inp=i['Extracted User Query']
    expected_output=i['Gold Responses']

    metric = AnswerRelevancyMetric(
        threshold=0,
        model="gpt-4o",
        include_reason=True,
        async_mode=False
    )
    test_case = LLMTestCase(
        input=inp,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected_output
    )

    metric.measure(test_case)
    print(metric.score)
    print(metric.reason)
    metrics.append(metric.score)

    # or evaluate test cases in bulk
    evaluate([test_case], [metric])
print(metrics)





