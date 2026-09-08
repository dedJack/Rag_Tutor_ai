import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib 

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


jsons = os.listdir("merge_chunks")
my_dicts = []
chunk_id = 1
for json_file in jsons:
    print(F"FILENAME:-{json_file}")
    with open(os.path.join('merge_chunks',json_file),"r", encoding='utf-8') as file:
        content = json.load(file)

    texts = [chunk['text'] for chunk in content['chunks']]

    print(f"\nFile: {json_file}")
    print(f"Chunks: {len(texts)}")
    print(f"Characters: {sum(len(x) for x in texts)}")

    embedding = create_chunks(texts)

    # If somefile is not embedding or response.status_code is not 200
    if not embedding:
        print("❌ Embedding failed")
        continue

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embedding[i]

        chunk_id += 1
        my_dicts.append(chunk)


# print(my_dicts)
df = pd.DataFrame.from_records(my_dicts)

joblib.dump(df,"embeddings.joblib")
print("All embeddings are done👍")
# df.to_csv("embeddings.csv", index=False)
# print(df)

