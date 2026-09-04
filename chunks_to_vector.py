import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def create_chunks(text):
    response = requests.post("http://localhost:11434/api/embed",json={
        "model":'bge-m3',
        "input":text,
    })

    if response.status_code != 200:
        print("Ollama response:")
        print(response.text)
        return []
    
    embedding = response.json()["embeddings"]
    return embedding


jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 1
for json_file in jsons:
    print(F"FILENAME:-{json_file}")
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)

    texts = [chunk['text'] for chunk in content['chunk']]

    print(f"\nFile: {json_file}")
    print(f"Chunks: {len(texts)}")
    print(f"Characters: {sum(len(x) for x in texts)}")

    embedding = create_chunks(texts)

    # If somefile is not embedding or response.status_code is not 200
    if not embedding:
        print("❌ Embedding failed")
        continue

    for i, chunk in enumerate(content['chunk']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embedding[i]

        chunk_id += 1
        my_dicts.append(chunk)
        if(i==5):
            break
    print(f"Completed: {json_file}")
    break


# print(my_dicts)
df = pd.DataFrame.from_records(my_dicts)

input_query = input("Ask a question for RAG = ")     #input query
question_embedding = create_chunks([input_query])[0]       # Creating embedding of input query

# print(np.vstack(df['embedding']).shape) #np.vstack allign the dimension vertically

similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()   #Finding cosine similarity between chunks
new_index = similarities.argsort()[::-1][0:3]
# print(similarities)
# print(similarities.argsort()[::-1][0:3])

new_df = df.loc[new_index]
print(new_df['text'])
# df.to_csv("embeddings.csv", index=False)
# print(df)

