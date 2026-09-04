from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
import numpy as np
import requests

def create_chunks(text):
    response = requests.post("http://localhost:11434/api/embed",json={ 
        "model":'bge-m3',
        "input":text
    })

    if(response.status_code!=200):
        print("Ollama response:")
        print(response.text)
        return []
    
    embedding = response.json()['embeddings']
    return embedding;

df = joblib.load("embeddings.joblib")
input_query = input("Ask a question for RAG = ")     #input query
question_embedding = create_chunks([input_query])[0]       # Creating embedding of input query

# print(np.vstack(df['embedding']).shape) #np.vstack allign the dimension vertically

similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()   #Finding cosine similarity between chunks
new_index = similarities.argsort()[::-1][0:3]
# print(similarities)
# print(similarities.argsort()[::-1][0:3])

new_df = df.loc[new_index]
print(new_df['text'])