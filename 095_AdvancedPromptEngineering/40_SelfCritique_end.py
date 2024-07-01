# %% packages
# related to: https://medium.com/@cobusgreyling/self-critique-llm-chain-using-langchain-smartllmchain-d67c42a4fa83
# run pip install langchain_experimental
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_experimental.smart_llm import SmartLLMChain

# %%
hard_question = "If it takes 5 workers 8 days to build a house, how long would it take 10 workers to build the same house?"
model_name = "gpt-3.5-turbo"
# %% baseline
import openai
from openai import OpenAI
import os

openai.api_key = os.environ["OPENAI_API_KEY"]
openai_client = OpenAI()
messages = [{"role": "user", "content": hard_question}]
response = openai_client.chat.completions.create(model=model_name, messages=messages)
content = response.choices[0].message.content
print(f"baseline zero-shot: {content}")

# %% import SmartLLMChain

prompt = PromptTemplate.from_template(hard_question)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=3, verbose=True)

# %%
chain.run({})
# %%
