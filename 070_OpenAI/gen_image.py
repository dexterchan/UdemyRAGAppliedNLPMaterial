# %% gen space probe
from openai import OpenAI

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="a small space probe landed on a distant planet remotely",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
image_url
# %% gen wheat field
from openai import OpenAI

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="We are in a golden wheat field which is ready for harvest in a sunny day. A harvestor is harvesting wheat. The sun should be higher",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
image_url
# %%
from openai import OpenAI
import os

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.images.generate(
    model="dall-e-3",
    prompt="an Asian sitting on an island waving for a Coast Guard ship to rescue him",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
image_url
# %%
from openai import OpenAI
import os

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.images.generate(
    model="dall-e-3",
    prompt="a coding boar coding in front of a laptop and have a coffee besides him",
    size="724x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
image_url
