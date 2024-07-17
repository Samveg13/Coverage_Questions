import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from datasets import Dataset 
from ragas.metrics import faithfulness,answer_relevancy,answer_correctness,answer_similarity,context_entity_recall,context_precision,ContextRelevancy,context_recall,context_entity_recall
from ragas import evaluate
import ast

# Initialize BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def process_csv_file(file_path):
    df = pd.read_csv(file_path)  # Adjust if necessary for your file format

    if 'AI_Response' not in df.columns or 'Gold Responses' not in df.columns or 'Gold Answer Present' not in df.columns:
        raise ValueError("The CSV file must contain 'generated', 'ground truth', and 'jkjk' columns.")
    
    # Filter rows where 'jkjk' is 'Yes'
    df_filtered = df[df['Gold Answer Present'] == 'Yes']
    df=df_filtered
    
    
    return df_filtered

def build_dataset(file_path):
    df=process_csv_file(file_path)
    data_samples = {'question': [], 'answer': [], 'ground_truth': [],'contexts':[]}
    
    for index, row in df.iterrows():
      
        
        data_samples['question'].append(row['Extracted User Query'])
        data_samples['answer'].append(row['colbert_response'])
        data_samples['ground_truth'].append(row['Gold Responses'])
        data_samples['contexts'].append([(row['col_source_pass'])])
    # print(data_samples['contexts'][0])
    dataset = Dataset.from_dict(data_samples)
    return dataset

def faithful(file_path):
    
        # data_samples['contexts'].append(contexts)
    dataset=build_dataset(file_path)
    score = evaluate(dataset,metrics=[context_entity_recall])
    fd=score.to_pandas()
    print("The context_entity_recall score is: "+ str(np.mean(fd['context_entity_recall'])))
    # print("The answer_correctness score is: "+ str(np.mean(fd['answer_correctness'])))
    print(type(score.to_pandas()))
    score.to_pandas().to_csv('eval.csv')
    # context_relevancy = ContextRelevancy()
    # results = context_relevancy.score(dataset)

    # print(results)
    return(score.to_pandas())


# Function to compute BERT embeddings
def get_bert_embeddings(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.cpu().numpy()

# Function to compute cosine similarity
def compute_cosine_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)[0][0]

# Load and process the CSV file


# Main function to compute metrics
def compute_metrics(file_path):
    df = process_csv_file(file_path)

    
    # Initialize metrics list
    similarities = []
    processed_rows = 0

    for index, row in df.iterrows():
        generated_text = row['AI_Response']
        ground_truth_text = row['Gold Responses']
        
        try:
            # Compute embeddings
            generated_embedding = get_bert_embeddings(generated_text, model, tokenizer, device)
            ground_truth_embedding = get_bert_embeddings(ground_truth_text, model, tokenizer, device)
        
            # Compute cosine similarity
            similarity = compute_cosine_similarity(generated_embedding, ground_truth_embedding)
            similarities.append(similarity)
            
            processed_rows += 1
        
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    # Convert lists to numpy arrays and compute average similarity
    similarities = np.array(similarities)
    print(similarities)
    average_similarity = similarities.mean() if len(similarities) > 0 else 0
    
    # Output results
    print(f"Total rows processed: {processed_rows}")
    print(f"Average cosine similarity: {average_similarity}")

if __name__ == "__main__":
    file_path = '/Users/samveg.shah/Desktop/Llama-index/data_finale.csv'  # Update this with the path to your CSV file
    print("The faithfullness score is:"+ str(faithful(file_path)))
    # compute_metrics(file_path)