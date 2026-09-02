import requests
import os
import json
import pandas as pd

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

    print(f"Completed: {json_file}")

# print(my_dicts)
df = pd.DataFrame.from_records(my_dicts)
df.to_csv("embeddings.csv", index=False)
# print(df)
