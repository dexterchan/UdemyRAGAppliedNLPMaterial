# %% packages
import os
import streamlit as st
import chromadb
import openai
from openai import OpenAI

openai.api_key = os.environ["OPENAI_API_KEY"]

# %% data prep
chroma_client = chromadb.PersistentClient(path="db")
chroma_collection = chroma_client.get_or_create_collection("ipcc")


# %%
def rag(query, n_results=5):
    res = chroma_collection.query(query_texts=[query], n_results=n_results)
    docs = res["documents"][0]
    joined_information = ";".join([f"{doc}" for doc in docs])
    messages = [
        {
            "role": "system",
            "content": """
                        You are a helpful expert on climate change.
                        Your users are asking questions about information contained in attached information.
                        Answer the user's question using only this provided information.
                        Please answer within 200 words.            
                        """,
        },
        {
            "role": "user",
            "content": f"Question: {query}. \n Information: {joined_information}",
        },
    ]
    openai_client = OpenAI()
    model = "gpt-3.5-turbo"
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content, docs


# %%
st.header("Climate Change Chatbot")

# text input field
user_query = st.text_input(
    label="",
    help="Ask a question about climate change",
    placeholder="What do you want to know about climate change?",
)

n_results: int = 5

rag_response, raw_docs = rag(query=user_query, n_results=n_results)
st.header("Raw Information")
raw_docs = ["", "", "", "", ""] if len(user_query) == 0 else raw_docs
for i, doc in enumerate(raw_docs):
    st.text(f"Raw Response {i}: {doc}")

st.header("RAG Response")
rag_response = "" if len(user_query) == 0 else rag_response
st.write(rag_response)
