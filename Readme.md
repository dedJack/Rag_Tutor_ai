# AI Tutor Platform (RETRIEVAL AUGMENTED GENERATION) Project

### How to Run Locally?
Clone the repository, create a virtual environment, and install the required dependencies from `requirements.txt`.

### NOTICE - 
This project currently uses open-source models for embeddings and LLM responses.

You can replace them with APIs such as OpenAI or OpenRouter by modifying the corresponding implementation.

## Steps:-
1. Move all your videos into `videos` folder
2. Run `video_to_audio.py` file, It will convert all your videos to audio. By using ffmpeg software.
3. Next step is to run ` create_chunks.py ` file, Whisper produces timestamped transcription segments, and your script stores them as JSON chunks.
4. Then run ` chunks_to_vector.py ` file , this will convert chunks into embeddings and store it in a Joblib file".
5. Last step is to run `processing_similarity.py`, This is the main part of our RAG system. In this file the embeddings are called from joblib memory and then cosine similarity is used to find out top related chunks according to user query. 
And the query and retrieved chunks are passed to the LLM as context and then the LLM generate the response.


# MODELS:-
This project uses open source whisper repository for Transcription
Link:- `pip install git+https://github.com/openai/whisper.git `

Also it uses Ollama models
Link:- `https://ollama.com`

## Folder structure:-
AI-tutor/
├── venv/
├── videos/
├── audios/
├── jsons/
├── create_chunks.py
├── chunks_to_vector.py
├── processing_similarity.py
├── video_to_audio.py
└── requirements.txt