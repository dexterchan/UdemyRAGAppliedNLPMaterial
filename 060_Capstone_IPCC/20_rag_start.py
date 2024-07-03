# %% packages
import os
import re
from pprint import pprint
from pypdf import PdfReader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_core.documents import Document
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
import openai
from openai import OpenAI

openai.api_key = os.environ["OPENAI_API_KEY"]

# %% Text Extraction
ipcc_pdf: str = "../data/IPCC_AR6_WGII_TechnicalSummary.pdf"
reader = PdfReader(ipcc_pdf)
ipc_texts = [page.extract_text() for page in reader.pages]


# %% Show first page
pprint(ipc_texts[0])
# %% filter out beginning and end of document
filtered_ipc_texts = ipc_texts[5:-4]
# %% remove all header / footer texts
# remove "Technical Summary" and the number before it
ipcc_wo_header_footer: list[str] = [
    re.sub(r"\d+\nTechnical Summary", "", text) for text in filtered_ipc_texts
]

# remove \nTS
ipcc_wo_header_footer = [re.sub(r"\nTS", " ", text) for text in ipcc_wo_header_footer]

# remove TS\n
ipcc_wo_header_footer = [re.sub(r"TS\n", " ", text) for text in ipcc_wo_header_footer]

# %% Convert list of string to Documents list
documents: list[Document] = [
    Document(page_content=t, metadata={"page": i})
    for i, t in enumerate(ipcc_wo_header_footer)
]
# %% Split Text
# the model card support 256 tokens, on the average each token taking
# 4 characters. therefore, chunk_size=1000
char_splitter = RecursiveCharacterTextSplitter(
    # Set parameters for splitting
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0.2,
    length_function=len,
)

text_documents = char_splitter.split_documents(documents=documents)
# %% Token Split
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0.2,
    tokens_per_chunk=256,
)

texts_token_splitted = []
for document in text_documents:
    try:
        texts_token_splitted.extend(token_splitter.split_text(document.page_content))
    except Exception as ex:
        print(f"{ex}: {document}")
        continue

# %% Show first chunk
texts_token_splitted[0]
# %% Vector Database
chroma_db = chromadb.PersistentClient(path="db")

embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

ipcc_collection = chroma_db.get_or_create_collection(
    name="ipcc", embedding_function=embedding_fn
)

# %% Add Documents
ids = [str(i) for i in range(len(texts_token_splitted))]

ipcc_collection.add(ids=ids, documents=texts_token_splitted)
# %% query DB
query = "What is the impact of climate change on the ocean?"
res = ipcc_collection.query(query_texts=[query], n_results=5)


# %% RAG


# %% Test RAG

# %%
