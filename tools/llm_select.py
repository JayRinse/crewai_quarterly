import json
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
from langchain_google_genai import GoogleGenerativeAI


def langchain_gemini_flash(
    api_path: str = "/home/johan/Code/secrets/google_api.json",
):

    with open(api_path, "r") as file:
        google_api = json.load(file)
    GOOGLE_API_KEY = google_api["api_key"]

    print(f"The online model has been selected 💸")
    llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
    # llm = Gemini(model="models/gemini-1.5-flash-latest", api_key=GOOGLE_API_KEY)
    # embed_model = GeminiEmbedding(
    #     model_name="models/embedding-001", api_key=GOOGLE_API_KEY
    # )
    # token_counter = TokenCountingHandler(
    #         tokenizer=tiktoken.get_encoding(llm).encode
    #     )
    # callback_manager = CallbackManager([token_counter])
    # Settings.callback_manager = callback_manager
    # Settings.llm = llm
    # Settings.embed_model = embed_model

    return llm #, embed_model

def gemini_flash(
    api_path: str = "/home/johan/Code/secrets/google_api.json",
):

    with open(api_path, "r") as file:
        google_api = json.load(file)
    GOOGLE_API_KEY = google_api["api_key"]

    print(f"The online model has been selected 💸")
    llm = Gemini(
        model="models/gemini-1.5-flash-latest", 
        api_key=GOOGLE_API_KEY,
        # generation_config={"termperature": 0.1} 
        )
    embed_model = GeminiEmbedding(
        model_name="models/embedding-001", api_key=GOOGLE_API_KEY
    )
    # token_counter = TokenCountingHandler(
    #         tokenizer=tiktoken.get_encoding(llm).encode
    #     )
    # callback_manager = CallbackManager([token_counter])
    # Settings.callback_manager = callback_manager
    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model

def gemma2b(
        
):
    print(f"The local model has been selected 💻")
    llm = Ollama(model='gemma:2b', request_timeout=6*60)
    embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-base-en-v1.5')
    Settings.llm = llm
    Settings.embed_model = embed_model

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("No GPU available, using CPU instead.")
        device = torch.device("cpu")
    
    return llm, embed_model