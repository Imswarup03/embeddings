import json
import boto3
import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

model_id = "cohere.embed-english-v3"
region_name = 'us-east-1'
accept = "application/json"
content_type = "application/json"
input_type = "search_query"

bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    aws_access_key_id=os.environ.get('aws_access_key_id'),
    aws_secret_access_key=os.environ.get('aws_secret_access_key'),
    region_name=os.environ.get('region_name')
)

def generate_embeddings(text):
    body = json.dumps({
        "texts": [text],
        "input_type": input_type
    })

    print("Generating embeddings...")
    response = bedrock_client.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )

    response_body = json.loads(response.get('body').read())
    embeddings = response_body.get("embeddings")

    return embeddings

def result_embeddings(query, collection_name):
    client = chromadb.PersistentClient(path='db/')
    collection = client.get_collection(name=collection_name)
    query_embedding = generate_embeddings(query)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )
    return results

def process_results_with_llm(query, chroma_results):
    # Extract the first two results safely
    top_two_results = []
    ids = chroma_results.get('ids', [[]])[0]
    distances = chroma_results.get('distances', [[]])[0]
    metadatas = chroma_results.get('metadatas', [[]])[0]
    documents = chroma_results.get('documents', [[]])[0]

    # Check if there are at least two results
    if len(ids) < 2:
        print("Warning: Less than two IDs available.")

    for i in range(min(2, len(ids))):
        result = {
            "id": ids[i] if i < len(ids) else None,
            "distance": distances[i] if i < len(distances) else None,
            "metadata": metadatas[i] if i < len(metadatas) else None,
            "document": documents[i] if i < len(documents) else None
        }
        top_two_results.append(result)

    # Prepare the prompt for the LLM
    prompt = f"""Given the following user question and two relevant pieces of information, provide an accurate and concise answer.

User Question: {query}

Relevant Information 1:
{json.dumps(top_two_results[0], indent=2)}

Relevant Information 2:
{json.dumps(top_two_results[1], indent=2) if len(top_two_results) > 1 else "Not enough information to provide a second relevant piece."}

Based on the above information, please provide a clear and accurate answer to the user's question. If the information is insufficient to answer the question fully, please state so and provide the best possible answer with the available information."""

    # Invoke the LLM model (using Claude v2 as an example)
    response = bedrock_client.invoke_model(
        body=json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 500,
            "temperature": 0.7,
            "top_p": 0.95,
        }),
        modelId='anthropic.claude-v2',
        accept='application/json',
        contentType='application/json'
    )

    # Extract and return the LLM's response
    response_body = json.loads(response.get('body').read())
    return response_body.get('completion')

# Main execution
if __name__ == "__main__":
    text = "susrut eye"
    chroma_collection_name = 'uservalues1'

    print("Querying Chroma collection...")
    chroma_results = result_embeddings(text, chroma_collection_name)
    print("Chroma query results:", chroma_results)

    print("\nProcessing results with LLM...")
    llm_response = process_results_with_llm(text, chroma_results)
    print("LLM Response:", llm_response)
