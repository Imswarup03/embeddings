import boto3
import json
import os
from botocore.config import Config
import chromadb
import shutil
import sys
from dotenv import load_dotenv
__import__('pysqlite3')


model_id = "cohere.embed-english-v3"
region_name= "us-east-1"
normalize= True
accept = "application/json"
content_type= 'application/json'
input_type= "search_query"
embedding_type= ["int8", "float"]
my_config = Config(
    connect_timeout = 60*3,
    read_timeout = 60*3
)

bedrock_session = boto3.Session(region_name= region_name)
bedrock_runtime = bedrock_session.client('bedrock-runtime',
                                        aws_access_key_id= os.environ.get('aws_access_key_id'),
                                        aws_secret_access_key=os.environ.get('aws_secret_access_key_id'),
                                        config= my_config)

def generate_embeddings(text):
    try:
        body = json.dumps({
            "texts": [text],
            "input_type": input_type
        })
        response = bedrock_runtime.invoke_model(
            body= body,
            modelId = model_id,
            accept = accept,
            contentType= content_type
        )
        response_body  = json.loads(response.get('body').read())
        embeddings = response_body.get('embeddings')

        return embeddings
    except Exception as e:
        return None
    



def result_embeddings(query, collection_name):
    client = chromadb.PersistentClient(path= 'db/')
    collection = client.get_collection(name= collection_name)
    query_embedding = generate_embeddings(query)
    results = collection.query(query_embedding = query_embedding, n_results = 5)

    return results


# def create_chunk(file_path):
#     sam_list= []

#     with open(file_path, 'r') as file:
        
