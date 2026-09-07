import json

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

def streamResponse(prompt):
    response = requests.post("http://localhost:11434/api/generate",json={ 
        "model":'llama3.2',
        "prompt":prompt,
        "stream":False
    })

    if(response.status_code!=200):
        print("Ollama response:")
        print(response.text)
        return []
    
    result = response.json()
    # print(result)
    return result;

df = joblib.load("embeddings.joblib")
input_query = input("Ask a question for RAG = ")     #input query
question_embedding = create_chunks([input_query])[0]       # Creating embedding of input query

# print(np.vstack(df['embedding']).shape) #np.vstack allign the dimension vertically
top_result=30
similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()   #Finding cosine similarity between chunks
new_index = similarities.argsort()[::-1][0:top_result]
# print(similarities)
# print(similarities.argsort()[::-1][0:3])

new_df = df.loc[new_index].reset_index(drop=True)
# print(new_df[['video_no','title','text']])

prompt = f"""
You are an AI tutor. Answer the user's question using only the provided context.

Context:
{new_df[['video_no','title','start','end','text']].to_dict(orient="records")}

Question:
{input_query}

Instructions:
- Use only the information from the context.
- Do not make up information.
- If the answer is not present in the context, say: "I don't have enough information in the provided context.", and tell them to watch video of most related content from course
- Give a clear and concise answer.
"""

response = streamResponse(prompt)
r = response['response']
print(r)
with open("response.json","w") as f:
    json.dump(r, f)

# print(prompt)
# for index, item in new_df.iterrows():
#     print(index , item['video_no'],item['title'], item['text'], item['start'],item['end'])