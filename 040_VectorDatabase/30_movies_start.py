# %% packages
import re
import pandas as pd
import seaborn as sns
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pprint import pprint


# %% max_length
def max_word_count(txt_list: list):
    max_length = 0
    for txt in txt_list:
        word_count = len(re.findall(r"\w+", txt))
        if word_count > max_length:
            max_length = word_count
    return f"Max Word Count: {max_length} words"


# %% Sentence splitter
# chroma default sentence model "all-MiniLM-L6-v2"
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# max input length: 256 characters
model_max_chunk_length = 256
token_splitter = SentenceTransformersTokenTextSplitter(
    tokens_per_chunk=model_max_chunk_length,
    model_name="all-MiniLM-L6-v2",
    chunk_overlap=0,
)

# %% Data Import
data_file = "../data/movies.csv"
df = pd.read_csv(data_file, parse_dates=["release_date"])
df.head()
# %% filter movies for missing title or overwiew
df_movies_filtered = df.dropna(subset=["title", "overview"])
# filter for unique ids

df_movies_filtered.drop_duplicates(subset=["id"], inplace=True)
# %%
# Check the shape here
df_movies_filtered.shape

# %%
max_word_count(df_movies_filtered["overview"])
# %% Word Distribution
description_len = []
for txt in df_movies_filtered.loc[:, "overview"]:
    description_len.append(len(re.findall(r"\w+", txt)))

# %% visualize token distribution
sns.histplot(description_len)
# %% embedding function
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# %%
# chroma_db = chromadb.Client()  # on the fly
# persistent
chroma_db = chromadb.PersistentClient("database.db")

# %% List Collections
chroma_db.list_collections()

# %% Get / Create Collection
chroma_collection = chroma_db.get_or_create_collection(
    name="moveies", embedding_function=embedding_fn
)

# %% add all tokens to collection
ids = df_movies_filtered["id"].values.astype(str).tolist()
documents = df_movies_filtered["overview"].values.tolist()
titles = df_movies_filtered["title"].values.tolist()
metadatas = [{"title": t} for t in titles]
# %% Add movies to vector DB
# add documents in batches
BATCH_SIZE: int = 5000
for i in range(0, len(ids), BATCH_SIZE):
    begin, end = i, i + BATCH_SIZE
    print(f"Processing batch {begin}:{end}")
    chroma_collection.add(
        ids=ids[begin:end],
        metadatas=metadatas[begin:end],
        documents=documents[begin:end],
    )


# %% count of documents in collection
len(chroma_collection.get()["ids"])


# %% Function to get title
def get_title_by_description(query_text: str) -> list[str]:
    n_best = 3
    res = chroma_collection.query(query_texts=[query_text], n_results=n_best)

    return [res["metadatas"][0][i]["title"] for i in range(n_best)]


# %% Test the function
get_title_by_description(query_text="super big lizard")

# %%
