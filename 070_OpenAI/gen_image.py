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
    prompt="a coding boar sitting in front of a macbook pro. He has a cup of coffee and a plate of cheese on the table",
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
    prompt="Three cartoon robot dinasaurs walking in a garden. The garden is covered with grasses. We have trees, flowers and bees in the garden. A rainbow is also in the sky",
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
    prompt="cartoon gozilla atomic breath",
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
    prompt="cartoon gozilla atomic breath",
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
    prompt="Design a poster without words: Not to reply scam sms and click the hyperlink. Never tell personal details and transfer any money",
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
    prompt="切勿回覆詐騙短訊，不要打開超連結。不要披露個人資料及轉帳金錢",
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
    prompt="The theme of starship tropper. An army platoon are shooting the Big bugs. The army platoon is losing ground",
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
    prompt="The theme of starship tropper. A team of robots are killing the big bugs",
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
    prompt="A group of Asian primary school students participating in Swimming gala in cartoon style. six of the student compete in the 50m swimming race. Each of them wearing swimming goggles",
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
    prompt="This is the year of snake. Design a poster in Traditional Chinese to celebrate the year of snake. The snake should be cute. The banner should have '蛇年大吉'. ",
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
    prompt="a cute snake celebrating lunar new year. Firework in the background.",
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
    prompt="Marvel's thanos (bold head) ready to lead the world",
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
    prompt="a cartoon superhero, cute spiderBoar (looks like spiderham) croaching on a building",
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
    prompt="a cartoon gollum from lord of the ring, citque his thought on a problem",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
image_url