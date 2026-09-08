import json
import os
import math

files = os.listdir("jsons")
n=5
for file_name in files:
    # for chunk in chunks:
    if file_name.endswith('.json'):
        file_path = os.path.join("jsons", file_name)
        with open(file_path,"r") as f:
            data = json.load(f)
            merge_chunks = []
            num_chunks = len(data['chunk'])
            num_groups = math.ceil(num_chunks/n)

            for i in range(num_groups):
                start_idx = i*n
                end_idx = min((1+i)*n, num_chunks)

                chunk_group = data['chunk'][start_idx:end_idx]

                merge_chunks.append({'video_no': chunk_group[0]['video_no'], 'title': chunk_group[0]['title'], 'start': chunk_group[0]['start'], 'end': chunk_group[-1]['end'], 'text': " ".join(c['text'] for c in chunk_group)})

            os.makedirs('merge_chunks',exist_ok=True)
            with open(os.path.join('merge_chunks',file_name),"w",encoding='utf-8') as json_data:
                json.dump({'chunks': merge_chunks, "text":data['text']},json_data, indent=4)

print("Merging of chunks completed")

            