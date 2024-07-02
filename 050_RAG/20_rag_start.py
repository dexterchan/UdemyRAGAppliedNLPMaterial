# %% packages
import os
from pprint import pprint
import chromadb
import openai
from openai import OpenAI

openai.api_key = os.environ["OPENAI_API_KEY"]
# chroma_db = chromadb.Client()  # on the fly
chroma_db = chromadb.PersistentClient(path="./database.db")

# %% List Collections
chroma_db.list_collections()

# %% Get / Create Collection
movies_collection = chroma_db.get_or_create_collection(name="movies")


# %% count of documents in collection
len(movies_collection.get()["ids"])


# %% Query Function
def get_query_results(query_text: str, n_results: int = 5):
    res = movies_collection.query(query_texts=[query_text], n_results=n_results)
    docs = res["documents"][0]

    titles = [item["title"] for item in res["metadatas"][0]]
    res_string = ";".join(
        [f"{title}:({description})" for title, description in zip(titles, docs)]
    )

    return res_string


# %% RAG setup
system_role_definition: str = """
You are an expert in moveis.
Users ask you question about movies.
You will get a user question and relevant information.
Relevant information is structured like:
movie title:movie plot; .... 
with ";" separating each movie information.

Please answer the question ONLY using the information provided.
"""

user_query = "What are the names of the movies and their plot where {relevant_info} ?"


def generate_llm_prompt(query_text: str, n_results: int = 5) -> list[dict]:
    relevant_info = get_query_results(query_text=query_text, n_results=n_results)

    messages = [
        {"role": "system", "content": system_role_definition},
        {"role": "user", "content": user_query.format(relevant_info=relevant_info)},
    ]
    return messages


# %% Pass Vector DB response to RAG
openai_client = OpenAI()
model = "gpt-3.5-turbo"
query = "huge lizard"
openai_response = openai_client.chat.completions.create(
    model=model,
    messages=generate_llm_prompt(query),
    temperature=0.2,
)


# %% Response from RAG
openai_response.choices[0].message.content

# %%


# %% bundle everything in a function
def rag(query_text: str, n_results: int = 5) -> list[str]:
    messages: list[dict] = generate_llm_prompt(
        query_text=query_text, n_results=n_results
    )

    openai_response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    return openai_response.choices[0].message.content.split("\n")


# %% Response from Vector DB
print("Response from Vector DB")
print("-------------------------------------------------")
query = "hero save the earth"
res = rag(query_text=query, n_results=3)

pprint(res)
# %%

# %%
print("Response from RAG")
print("-------------------------------------------------")
