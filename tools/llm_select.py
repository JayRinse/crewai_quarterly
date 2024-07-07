import json
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings


def gemini_flash(
    api_path: str = "/home/johan/Code/secrets/google_api.json",
):

    with open(api_path, "r") as file:
        google_api = json.load(file)
    GOOGLE_API_KEY = google_api["api_key"]

    print(f"The online model has been selected 💸")
    llm = Gemini(model='models/gemini-1.5-flash-latest', api_key=GOOGLE_API_KEY)
    embed_model = GeminiEmbedding(model_name='models/embedding-001', api_key=GOOGLE_API_KEY)
    # token_counter = TokenCountingHandler(
    #         tokenizer=tiktoken.get_encoding(llm).encode
    #     )
    # callback_manager = CallbackManager([token_counter])
    # Settings.callback_manager = callback_manager
    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model