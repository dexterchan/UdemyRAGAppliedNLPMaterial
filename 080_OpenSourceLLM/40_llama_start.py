# LLM runner will be llama 3 supported by https://hub.docker.com/r/ollama/ollama
# trigger Ollama : https://ollama.com/library/llama3
# docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
# docker exec -it ollama ollama pull llama3 
# %% packages
import os
import chromadb

# %% data prep
chroma_client = chromadb.PersistentClient(path="db")
chroma_collection = chroma_client.get_or_create_collection("ipcc")


# %%
import requests
import json
import time


def rag(query, n_results=5):
    time_start = time.time()
    res = chroma_collection.query(query_texts=[query], n_results=n_results)
    docs = res["documents"][0]
    joined_information = ";".join([f"{doc}" for doc in docs])

    system_prompt: str = """
                        You are a helpful expert on climate change.
                        Your users are asking questions about information contained in attached information.
                        Answer the user's question using only this provided information.
                        Please answer within 200 words.            
                        """
    user_prompt: str = f"Question: {query}. \n Information: {joined_information}"

    prompt: str = f"[INST]<<SYS>>{system_prompt}<</SYS>>{user_prompt}[/INST]"

    # calling ollama
    url = "http://localhost:11434/api/generate"

    payload = json.dumps({"model": "llama3", "prompt": prompt, "stream": False})
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)

    time_end = time.time()

    print(f"elapsed time: {time_end-time_start}")
    return response.json()["response"]


# %% calling rag
from pprint import pprint

response_text = rag("What is the climate change impact to our seafood supply?")
pprint(response_text)
# %%
