import os
import whisper
import json


model = whisper.load_model("large-v2")
audios = os.listdir("audios")

for audio in audios:
    result = model.transcribe(audio=f"audios/{audio}",
                          language='hi',
                          task="translate",
                          word_timestamps=False)

    chunk = []
    video_no = f"{audio.split("_")[0]}"
    title = f"{audio.split("_")[1].strip(".mp3")}"

    for segment in result["segments"]:
        chunk.append({"id":segment['id'],"video_no": video_no , "title":title, "start":segment['start'],"end":segment['end'],"text":segment['text']})

    chunk_metadata = {"chunk":chunk, "text":result['text']}

    with open(f"jsons/{audio}.json", "w") as f:
        json.dump(chunk_metadata, f)